#!/usr/bin/env python3
# batch_asr_pipeline.py
import os, sys, re, json, glob, argparse, subprocess, shutil, tempfile, torch, functools, gc, math
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import soundfile as sf
from faster_whisper import WhisperModel

# ----------------- GPU prefs -----------------
if torch.cuda.is_available():
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# ----------------- utils -----------------
def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def dbfs(x: float) -> float:
    return 20.0 * math.log10(max(x, 1e-12))

def undb(db: float) -> float:
    return 10.0 ** (db / 20.0)

def run(cmd: List[str], cwd: Path | None = None, timeout: int = 1800):
    subprocess.run(cmd, check=True, cwd=(str(cwd) if cwd else None), timeout=timeout)

@functools.lru_cache(maxsize=1)
def get_whisper(model_name, device, compute):
    return WhisperModel(model_name, device=device, compute_type=compute)

@functools.lru_cache(maxsize=1)
def get_pyannote():
    from pyannote.audio import Pipeline
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return pipe

def ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def find_first(root: Path, name: str) -> Path:
    for r,_,fs in os.walk(root):
        for f in fs:
            if f.lower()==name.lower():
                return Path(r)/f
    raise FileNotFoundError(name)

def load_mono(path: Path) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), always_2d=False)
    if getattr(y, "ndim", 1) == 2:
        y = y.mean(axis=1)
    return y.astype(np.float32), sr

def fmt_ts(t: float) -> str:
    ms = int(round((t - int(t)) * 1000))
    h  = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

# ----------------- audio analysis → adaptive knobs -----------------
def analyze_audio(path: Path) -> Dict[str, Any]:
    x, sr = load_mono(path)
    n = len(x)
    dur = n / float(sr) if sr else 0.0

    peak = float(np.max(np.abs(x)) + 1e-12)
    rms  = float(np.sqrt(np.mean(x**2)) + 1e-12)
    dc   = float(np.mean(x))
    clip_frac = float(np.mean(np.abs(x) > 0.999))

    # frame RMS for noise estimate
    win = max(1, int(0.05 * sr))  # 50 ms
    hop = win
    if n < win: frame_rms = np.array([rms], dtype=np.float32)
    else:
        frames = [x[i:i+win] for i in range(0, n-win+1, hop)]
        frame_rms = np.array([np.sqrt(np.mean(f*f) + 1e-12) for f in frames], dtype=np.float32)
    noise_rms = float(np.percentile(frame_rms, 10))
    snr_db = dbfs(rms) - dbfs(noise_rms)

    # rough spectrum
    # take up to first 120 seconds for speed
    max_samps = int(min(n, sr * 120))
    xf = x[:max_samps]
    # apply Hann to reduce spectral leakage
    w = np.hanning(len(xf)) if len(xf) > 2048 else np.ones_like(xf)
    spec = np.fft.rfft(xf * w, n=len(xf))
    psd = (np.abs(spec)**2).astype(np.float64)
    freqs = np.fft.rfftfreq(len(xf), d=1.0/sr) if sr else np.linspace(0, 1, len(psd))

    def band_energy(lo, hi):
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if idx.size == 0: return 0.0
        return float(np.sum(psd[idx]))

    E_total = float(np.sum(psd) + 1e-18)
    E_0_80   = band_energy(0, 80)
    E_80_300 = band_energy(80, 300)
    E_300_3k = band_energy(300, 3000)
    E_3k_6k  = band_energy(3000, 6000)
    E_6k_9k  = band_energy(6000, 9000)
    E_9k_12k = band_energy(9000, 12000)

    rumble_ratio = (E_0_80 / E_total)
    mid_ratio    = (E_300_3k / E_total)
    hiss_ratio   = ((E_6k_9k + E_9k_12k) / E_total)
    sibilance_ratio = (E_6k_9k / (E_300_3k + 1e-18))

    # adaptive knobs
    rms_db = dbfs(rms)
    noise_db = dbfs(noise_rms)

    # dynaudnorm max gain g
    if rms_db < -35: dyn_g = 14
    elif rms_db < -30: dyn_g = 11
    elif rms_db < -24: dyn_g = 8
    else: dyn_g = 5

    # high-pass
    if rumble_ratio > 0.05 or abs(dc) > 0.01:
        hp = 120
    elif rumble_ratio > 0.02:
        hp = 100
    else:
        hp = 80

    # low-pass to tame hiss
    if hiss_ratio > 0.18:
        lp = 8000
    elif hiss_ratio > 0.12:
        lp = 8800
    else:
        lp = 9500

    # arnndn mix by SNR
    if snr_db < 5: rn_mix = 0.75
    elif snr_db < 10: rn_mix = 0.65
    elif snr_db < 15: rn_mix = 0.55
    elif snr_db < 20: rn_mix = 0.45
    elif snr_db < 25: rn_mix = 0.35
    else: rn_mix = 0.25

    # afftdn strength
    if snr_db < 6: afftdn_nr = 18
    elif snr_db < 12: afftdn_nr = 16
    elif snr_db < 20: afftdn_nr = 14
    else: afftdn_nr = 12

    # de-esser intensity
    if sibilance_ratio > 0.22: deess_i = 0.20
    elif sibilance_ratio > 0.15: deess_i = 0.16
    else: deess_i = 0.12

    # gate threshold near noise floor + safety
    thr_db = clamp(noise_db + 6.0, -60.0, -30.0)
    agate_thr = clamp(undb(thr_db), 0.001, 0.02)

    # compressor threshold and makeup
    target_db = -18.0
    comp_th = -28.0
    if rms_db < -30: comp_th = -35.0
    elif rms_db < -24: comp_th = -30.0
    elif rms_db > -18: comp_th = -24.0
    makeup = clamp(target_db - rms_db, 3.0, 10.0)

    # low EQ cut scaling for mud
    if (E_80_300 / (E_300_3k + 1e-18)) > 0.9:
        low_cut_scale = 1.0  # strong cuts
    elif (E_80_300 / (E_300_3k + 1e-18)) > 0.6:
        low_cut_scale = 0.7
    else:
        low_cut_scale = 0.4

    # presence boost scaling
    if mid_ratio < 0.20:
        presence_scale = 1.4
    elif mid_ratio < 0.30:
        presence_scale = 1.15
    else:
        presence_scale = 1.0

    return dict(
        sr=sr, dur=dur, rms_db=rms_db, peak_db=dbfs(peak), crest=(peak/(rms+1e-12)),
        dc=dc, clip_frac=clip_frac, noise_db=noise_db, snr_db=snr_db,
        rumble_ratio=rumble_ratio, hiss_ratio=hiss_ratio, mid_ratio=mid_ratio,
        hp=hp, lp=lp, dyn_g=dyn_g, rn_mix=rn_mix, afftdn_nr=afftdn_nr,
        deess_i=deess_i, agate_thr=agate_thr, comp_th=comp_th, makeup=makeup,
        low_cut_scale=low_cut_scale, presence_scale=presence_scale
    )

# ----------------- role heuristics + text utils -----------------
QUESTION_WORDS = {"who","what","where","when","why","how","which","do","does","did","is","are","can","could","would","will","have","has","had"}
ANSWER_STARTS  = {"yes","yeah","yep","right","okay","ok","uh-huh","sure","i","we","it","so","well","no","not","my","the"}

def is_question(txt: str) -> bool:
    t = (txt or "").strip().lower()
    if t.endswith("?"): return True
    first = re.split(r"\W+", t, maxsplit=1)[0] if t else ""
    return first in QUESTION_WORDS

def is_answerish(txt: str) -> bool:
    t = (txt or "").strip().lower()
    first = re.split(r"\W+", t, maxsplit=1)[0] if t else ""
    return (first in ANSWER_STARTS) or (len(t.split()) >= 5 and not is_question(t))

def relabel_turns_by_content_with_times(turns):
    out=[]; expect_answer=False
    for seg in turns:
        if not seg: continue
        txt = " ".join(w["text"] for w in seg).strip()
        if not txt: continue
        s = float(seg[0]["start"]); e = float(seg[-1]["end"])
        q = is_question(txt); a = is_answerish(txt)
        if q:
            role="Interviewer"; expect_answer=True
        elif expect_answer and a:
            role="Interviewee"; expect_answer=False
        else:
            role = "Interviewer" if q else "Interviewee"
        out.append((role, txt, s, e))
    return out

def basic_cleanup_text(s: str) -> str:
    t = re.sub(r"\s+([,.!?;:])", r"\1", s.strip())
    if t and t[-1] not in ".?!": t += "."
    t = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1)+m.group(2).upper(), t)
    return t

def chunk_role_lines(role_lines, max_chars=12000, max_turns=180):
    chunks, cur, cur_len, cur_n = [], [], 0, 0
    for role, txt, s, e in role_lines:
        line = f"{role} [{fmt_ts(s)}–{fmt_ts(e)}]: {txt}".strip()
        add = len(line) + 2
        if cur and (cur_len + add > max_chars or cur_n >= max_turns):
            chunks.append("\n\n".join(cur)); cur=[]; cur_len=0; cur_n=0
        cur.append(line); cur_len += add; cur_n += 1
    if cur: chunks.append("\n\n".join(cur))
    return chunks

def cleanup_workdir(work_dir: Path, keep: List[Path]):
    keep_set = {p.resolve() for p in keep}
    for root, dirs, files in os.walk(work_dir, topdown=False):
        for name in files:
            p = Path(root)/name
            if p.resolve() not in keep_set:
                try: p.unlink()
                except Exception: pass
        for name in dirs:
            d = Path(root)/name
            try: d.rmdir()
            except Exception: pass

# ----------------- FFmpeg wrappers -----------------
def ff_with_filter_script(in_wav: str, out_wav: str, graph: str, ar: str|None=None, ac: str|None=None, cwd: Path|None=None):
    with tempfile.NamedTemporaryFile("w", suffix=".ffgraph", delete=False, dir=(str(cwd) if cwd else None)) as tmp:
        tmp.write(graph)
        graph_path = tmp.name if not cwd else Path(tmp.name).name
    try:
        cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error","-i", in_wav, "-filter_script:a", graph_path]
        if ar: cmd += ["-ar", str(ar)]
        if ac: cmd += ["-ac", str(ac)]
        cmd += ["-c:a","pcm_s16le", out_wav]
        run(cmd, cwd=cwd, timeout=900)
    finally:
        try:
            (Path(cwd)/graph_path if cwd else Path(graph_path)).unlink(missing_ok=True)
        except Exception:
            pass

def ff(in_wav: str, out_wav: str, af: str, ar="16000", ac="1", cwd: Path|None=None):
    cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error","-i", in_wav]
    if ar: cmd += ["-ar", ar]
    if ac: cmd += ["-ac", ac]
    cmd += ["-af", af, "-c:a","pcm_s16le", out_wav]
    run(cmd, cwd=cwd, timeout=900)

def ff_has_arnndn() -> bool:
    try:
        out = subprocess.run(["ffmpeg","-hide_banner","-filters"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False).stdout.lower()
        return " arnndn " in out
    except Exception:
        return False

def arnndn_option_name() -> str:
    try:
        out = subprocess.run(["ffmpeg","-hide_banner","-h","filter=arnndn"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False).stdout.lower()
        if "model " in out or "model=" in out:
            return "model"
    except Exception:
        pass
    return "m"

# ----------------- adaptive FFmpeg graphs -----------------
def pre_chain_graph(metrics: Dict[str, Any]) -> Tuple[str, str, str]:
    # returns three parts for pre0, pre1, pre2
    # pre0: adaptive dynaudnorm + dc correction
    dc_shift = clamp(-metrics["dc"], -0.05, 0.05)
    pre0 = f"dynaudnorm=f=150:g={int(metrics['dyn_g'])}:m=10,dcshift=shift={dc_shift:.6f}"
    # pre1: low-cut EQs scaled
    s = float(metrics["low_cut_scale"])
    eqs = [
        (60,  -12*s),
        (120, -9*s),
        (240, -7*s),
        (360, -6*s),
        (480, -5*s),
    ]
    pre1 = ", ".join([f"equalizer=f={f}:t=q:w=2:g={g:.2f}" for f,g in eqs])
    # pre2: adaptive high-pass
    pre2 = f"highpass=f={int(metrics['hp'])}"
    return pre0, pre1, pre2

def vocal_stage1_graph(metrics: Dict[str, Any], model_key: str | None) -> str:
    hp = int(metrics["hp"])
    lp = int(metrics["lp"])
    rn = clamp(float(metrics["rn_mix"]), 0.15, 0.85)
    nr = int(metrics["afftdn_nr"])
    if model_key:
        prefix = f"arnndn={model_key}:mix={rn:.2f},"
    else:
        prefix = ""
    return (f"{prefix}"
            f"afftdn=nr={nr}:nt=w:om=o,highpass=f={hp},lowpass=f={lp}")

def vocal_stage2_graph(metrics: Dict[str, Any]) -> str:
    s = float(metrics["low_cut_scale"])
    eqs = [
        (60,  -12*s),
        (120, -9*s),
        (240, -7*s),
    ]
    return ", ".join([f"equalizer=f={f}:t=q:w=2:g={g:.2f}" for f,g in eqs])

def vocal_stage3_graph(metrics: Dict[str, Any]) -> str:
    p = float(metrics["presence_scale"])
    boosts = [
        (2200,  3.0*p),
        (3200,  2.0*p),
        (4500,  1.5*p),
    ]
    deess_i = float(metrics["deess_i"])
    eq = ", ".join([f"equalizer=f={f}:t=q:w=1.0:g={g:.2f}" for f,g in boosts])
    return f"{eq}, deesser=i={deess_i:.2f}"

def vocal_final_graph(metrics: Dict[str, Any]) -> str:
    thr = float(metrics["agate_thr"])
    comp_th = float(metrics["comp_th"])
    makeup = float(metrics["makeup"])
    return (f"agate=threshold={thr:.5f}:ratio=2:attack=5:release=80, "
            f"acompressor=threshold={comp_th:.0f}dB:ratio=2.5:attack=4:release=120:makeup={makeup:.1f}, "
            "alimiter=limit=0.98")

def vocal_boost_graph(metrics: Dict[str, Any]) -> str:
    # Presence and glue compression; keep adaptive minimalism
    p = float(metrics["presence_scale"])
    eqs = [
        (250,  2.0),
        (800,  3.0),
        (2000, 4.0*p),
        (3500, 3.0*p),
        (6000, 2.0*0.9),
    ]
    eq = ", ".join([f"equalizer=f={f}:t=q:w=1.0:g={g:.2f}" for f,g in eqs])
    return (f"{eq}, acompressor=threshold=-35dB:ratio=3:attack=2:release=150:makeup=6, alimiter=limit=0.98")

def residual_graph(metrics: Dict[str, Any], model_key: str | None) -> str:
    hp = int(metrics["hp"]) if metrics["hp"] >= 100 else 120
    lp = int(metrics["lp"])
    rn = clamp(float(metrics["rn_mix"]), 0.15, 0.85)
    nr = int(metrics["afftdn_nr"])
    arn = (f"arnndn={model_key}:mix={rn:.2f}," if model_key else "")
    return (
        f"highpass=f={hp},lowpass=f={lp}, "
        "equalizer=f=60:t=q:w=2:g=-12, equalizer=f=120:t=q:w=2:g=-9, equalizer=f=240:t=q:w=2:g=-7, "
        f"{arn}afftdn=nr={nr}:nt=w:om=o, "
        "equalizer=f=1800:t=q:w=1.0:g=2.5, equalizer=f=2800:t=q:w=1.0:g=2.0, "
        "acompressor=threshold=-40dB:ratio=2.2:attack=6:release=140:makeup=5, "
        "alimiter=limit=0.98"
    )

# ----------------- Demucs -----------------
def run_demucs(in_wav: Path, out_root: Path, device: str="auto", model="htdemucs", segment=10, overlap=0.25, shifts=2) -> Dict[str,Path]:
    dev = "cuda" if (device=="cuda" or (device=="auto" and shutil.which("nvidia-smi"))) else "cpu"
    ensure(out_root)
    cmd = [sys.executable,"-m","demucs.separate","-n",model,"-d",dev,"--jobs","1",
           "--segment",str(int(segment)),"--overlap",str(float(overlap)),"--shifts",str(int(shifts)),
           "--two-stems","vocals","--out",str(out_root), str(in_wav)]
    run(cmd, timeout=3600)
    vocals = find_first(out_root, "vocals.wav")
    no_vocals = find_first(out_root, "no_vocals.wav")
    return {"vocals": vocals, "no_vocals": no_vocals}

# ----------------- Diarization -----------------
def diarize_to_csv(in_wav: Path, out_csv: Path, num_speakers=2):
    import torch as _torch
    y, sr = load_mono(in_wav)
    waveform = _torch.from_numpy(y).unsqueeze(0)
    pipe = get_pyannote()
    pipe.to(_torch.device("cuda" if _torch.cuda.is_available() else "cpu"))
    diar = pipe({"waveform": waveform, "sample_rate": sr}, num_speakers=num_speakers)
    rows = [{"start": float(turn.start), "end": float(turn.end), "speaker": str(spk)}
            for turn, _, spk in diar.itertracks(yield_label=True)]
    pd.DataFrame(rows).sort_values(["start","end"]).to_csv(out_csv, index=False)

# ----------------- ASR -----------------
def decode_with_fallback(model, wav_path, t0, t1, vad_params,
                         temps=(0.0, 0.2, 0.4), lang="en"):
    with sf.SoundFile(str(wav_path)) as f:
        sr = f.samplerate; n  = len(f)
        i0 = max(0, int(t0 * sr)); i1 = min(n, int(t1 * sr))
        f.seek(i0)
        x = f.read(i1 - i0, dtype="float32", always_2d=False)
        if getattr(x, "ndim", 1) == 2: x = x.mean(axis=1)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        slice_path = tmp.name
    sf.write(slice_path, x, sr, subtype="PCM_16")
    best, best_score = None, -1e9
    try:
        for T in temps:
            segs, _ = model.transcribe(
                slice_path,
                language=lang,
                vad_filter=True,
                vad_parameters=vad_params,
                beam_size=8,
                best_of=8,
                temperature=T,
                no_speech_threshold=0.55,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.2,
                word_timestamps=True,
                condition_on_previous_text=False,
                length_penalty=0.8,
            )
            segs = list(segs)
            score = sum(getattr(s, "avg_logprob", -10.0) for s in segs) / max(1, len(segs))
            if score > best_score:
                best, best_score = segs, score
    finally:
        try: os.remove(slice_path)
        except Exception: pass
    out = []
    for s in (best or []):
        if not s.words: continue
        for w in s.words:
            if w.start is None or w.end is None: continue
            out.append({"start": float(w.start)+t0, "end": float(w.end)+t0, "text": w.word.strip()})
    return out

def transcribe_words_chunked(path: Path, model: WhisperModel,
                             chunk_sec=120.0, hop_sec=120.0,
                             vad_params=dict(threshold=0.30,
                                             min_speech_duration_ms=150,
                                             min_silence_duration_ms=800,
                                             speech_pad_ms=120)):
    with sf.SoundFile(str(path)) as f:
        dur = len(f) / float(f.samplerate)
    words=[]; t=0.0; last_text=""; repeat_run=0
    while t < dur:
        t0, t1 = t, min(dur, t + chunk_sec)
        w_chunk = decode_with_fallback(model, str(path), t0, t1, vad_params)
        this_text = " ".join(w["text"] for w in w_chunk).strip()
        if this_text and this_text == last_text:
            repeat_run += 1
            if repeat_run >= 2:
                t = t1 + 10.0
                continue
        else:
            repeat_run = 0
        last_text = this_text
        words.extend(w_chunk)
        t += hop_sec
    # drop repeated 6-grams
    def normtok(x): return re.sub(r"\W+", "", x.lower())
    buf=[]; out=[]; last6=None; rep=0
    for w in words:
        buf.append(normtok(w["text"]))
        if len(buf) >= 6:
            cur6 = tuple(buf[-6:])
            if cur6 == last6:
                rep += 1
                if rep >= 2: continue
            else:
                rep=0; last6=cur6
        out.append(w)
    return out

# ----------------- grouping + text -----------------
def load_segments_csv(path: Path):
    df = pd.read_csv(path)
    keep = [c for c in ["start","end","speaker"] if c in df.columns]
    if len(keep) < 3: raise SystemExit("segments.csv must have start,end,speaker.")
    df = df[keep].dropna().copy()
    df["start"]=df["start"].astype(float); df["end"]=df["end"].astype(float); df["speaker"]=df["speaker"].astype(str)
    df = df[df["end"]>df["start"]].sort_values(["start","end"])
    return df.to_dict("records")

def assign(words: List[Dict[str,Any]], spans: List[Dict[str,Any]]):
    def lab(t):
        for s in spans:
            if s["start"] <= t < s["end"]:
                return s["speaker"]
        return "SPK0"
    return [{**w, "speaker": lab(0.5*(w["start"]+w["end"]))} for w in words]

def estimate_max_gap(labeled, mult=3.0, lo=0.8, hi=2.5):
    gaps=[max(0.0, b["start"]-a["end"]) for a,b in zip(labeled, labeled[1:]) if a["speaker"]==b["speaker"]]
    if not gaps: return lo
    med=float(np.median(gaps))
    return float(min(hi, max(lo, med*mult)))

def group_turns(labeled, max_gap=0.7, speaker_switch_pad=0.0):
    if not labeled: return []
    labeled.sort(key=lambda x: x["start"])
    res, buf = [], [labeled[0]]
    for p,c in zip(labeled, labeled[1:]):
        dt = c["start"]-p["end"]
        switch = (c["speaker"]!=buf[-1]["speaker"]) and (dt>speaker_switch_pad)
        long_sil = dt>max_gap
        if switch or long_sil:
            res.append(buf); buf=[c]
        else:
            buf.append(c)
    res.append(buf); return res

def punct(words):
    s = " ".join(w["text"] for w in words).replace(" ,",",").replace(" .",".").strip()
    if s and s[-1] not in ".?!": s+="."
    return s

def fix_boundary_fragments(turns):
    HEADLIKE = {"what","when","where","who","why","how","which","that","and","but","so","because","the","a","an"}
    for i in range(len(turns) - 1):
        if not turns[i] or not turns[i+1]:
            continue
        last_tok = turns[i][-1].get("text","").strip().lower()
        last_tok = re.sub(r"[^\w']+$", "", last_tok)
        if last_tok in HEADLIKE:
            turns[i+1].insert(0, turns[i].pop())
    return [seg for seg in turns if seg]

# ----------------- LLM clean with timestamps -----------------
def llm_clean_chunks_ts(chunks, model: str, temperature=0.0, max_output_tokens=4000):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    instructions = (
        "You are a precise transcript editor. Preserve meaning and order.\n"
        "Each input line starts with a role and a timestamp span, e.g.:\n"
        "Interviewer [00:01:02.345–00:01:08.901]: text\n"
        "Rules:\n"
        " - Keep EVERY line. Same number of lines, same order.\n"
        " - Do NOT change or remove timestamps.\n"
        " - If a line ends with a lone headword (what/why/how/which/who/when/that/and/but/so/because/the/a/an)\n"
        "   and the next line clearly continues that sentence/question, FIX it by either:\n"
        "     (a) completing the first line with up to 5 short tokens and removing any duplicated headword at the\n"
        "         start of the next line, or\n"
        "     (b) moving the headword to the start of the next line if that yields better grammar.\n"
        "   Keep line count and timestamps unchanged.\n"
        " - Default questions to Interviewer; default declaratives/answers to Interviewee. Correct mislabeled roles.\n"
        " - Minimal paraphrase. No summaries. No extra lines."
    )
    pat = re.compile(r"^(Interviewer|Interviewee)\s*\[\d{2}:\d{2}:\d{2}\.\d{3}[–-]\d{2}:\d{2}:\d{2}\.\d{3}\]\s*:\s+.+$")
    out_parts=[]
    for raw_dialogue in chunks:
        prompt = "Edit under the rules. Keep timestamps EXACT and per-line alignment.\n\n" + raw_dialogue
        resp = client.responses.create(
            model=model or "gpt-4.1-mini",
            instructions=instructions,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        text = (resp.output_text or "").strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        kept = [ln for ln in lines if pat.match(ln)] or [ln.strip() for ln in raw_dialogue.splitlines() if ln.strip()]
        out_parts.append("\n\n".join(kept))
    return "\n\n".join(out_parts)

# ----------------- main pipeline per file -----------------
def process_one(wav_path: Path, args):
    base = wav_path.stem
    work = Path(args.out_dir)/base
    ensure(work)

    # stage RNNoise model in work dir
    local_model = work / "std.rnnn"
    if not local_model.exists():
        shutil.copyfile(args.rnnoise_model, local_model)

    use_rnndn = ff_has_arnndn()
    opt_name = arnndn_option_name()
    model_key = f"{opt_name}=std.rnnn" if use_rnndn else None

    # analyze input and derive knobs
    metrics = analyze_audio(wav_path)
    print(f"[ANALYZE] {base} | rms={metrics['rms_db']:.1f} dBFS, SNR={metrics['snr_db']:.1f} dB, "
          f"rumble={metrics['rumble_ratio']:.3f}, hiss={metrics['hiss_ratio']:.3f}", flush=True)

    # 1) preprocess (adaptive)
    pre0 = work/f"{base}_pre0.wav"
    pre1 = work/f"{base}_pre1.wav"
    pre2 = work/f"{base}_pre2.wav"
    pre3 = work/f"{base}_pre3.wav"
    g0,g1,g2 = pre_chain_graph(metrics)
    ff(str(wav_path), str(pre0), g0, ar=None, ac=None, cwd=work)
    ff(str(pre0), str(pre1), g1, ar=None, ac=None, cwd=work)
    ff(str(pre1), str(pre2), g2, ar=None, ac=None, cwd=work)

    if use_rnndn:
        rn_mix_pre = clamp(metrics["rn_mix"] * 0.9, 0.15, 0.85)
        graph_pre3 = f"aresample=48000,arnndn={model_key}:mix={rn_mix_pre:.2f},aresample=16000"
        ff_with_filter_script(str(pre2), str(pre3), graph_pre3, cwd=work)
    else:
        ff(str(pre2), str(pre3), "aresample=48000,aresample=16000", ar=None, ac=None, cwd=work)

    # 2) Demucs
    dem_in = work/f"{base}_demucs_in.wav"
    ff(str(pre3), str(dem_in), af="anull", ar="44100", ac="2", cwd=work)
    stems = run_demucs(dem_in, work/"separated", device=args.demucs_device,
                       model=args.demucs_model, segment=args.demucs_segment,
                       overlap=args.demucs_overlap, shifts=args.demucs_shifts)

    v1 = work/f"{base}_v1.wav"
    v2 = work/f"{base}_v2.wav"
    v3 = work/f"{base}_v3.wav"
    vclean = work/f"{base}_vocal_clean.wav"
    vboost = work/f"{base}_vocal_boost.wav"
    v16 = work/f"{base}_vocal_16k.wav"

    ff_with_filter_script(str(stems['vocals']), str(v1), vocal_stage1_graph(metrics, model_key), cwd=work)
    ff(str(v1), str(v2), vocal_stage2_graph(metrics), ar=None, ac=None, cwd=work)
    ff(str(v2), str(v3), vocal_stage3_graph(metrics), ar=None, ac=None, cwd=work)
    ff(str(v3), str(vclean), vocal_final_graph(metrics), ar=None, ac=None, cwd=work)
    ff(str(vclean), str(vboost), vocal_boost_graph(metrics), ar=None, ac=None, cwd=work)
    ff(str(vboost), str(v16), af="anull", ar="16000", ac="1", cwd=work)

    resid_clean = work/f"{base}_residual_clean.wav"
    resid16 = work/f"{base}_residual_16k.wav"
    ff_with_filter_script(str(stems["no_vocals"]), str(resid_clean), residual_graph(metrics, model_key), cwd=work)
    ff(str(resid_clean), str(resid16), af="anull", ar="16000", ac="1", cwd=work)

    # adaptive mix balance based on RMS
    def rms_db(path: Path) -> float:
        y, sr = load_mono(path)
        return dbfs(float(np.sqrt(np.mean(y*y)) + 1e-12))
    v_db = rms_db(v16); r_db = rms_db(resid16)
    # target residual ~8 dB below vocals
    want_delta = 8.0
    resid_gain = clamp(undb((v_db - r_db - want_delta)), 0.25, 1.20)
    mix16 = work/f"{base}_mix_16k.wav"
    filt = f"[0:a]volume=1.0[a0];[1:a]volume={resid_gain:.3f}[a1];[a0][a1]amix=inputs=2:normalize=0[out]"
    cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error",
           "-i", str(v16), "-i", str(resid16),
           "-filter_complex", filt, "-map","[out]","-ar","16000","-ac","1","-c:a","pcm_s16le", str(mix16)]
    run(cmd, cwd=work)

    # 3) diarize
    seg_csv = work/f"{base}_segments.csv"
    diarize_to_csv(mix16, seg_csv, num_speakers=2)

    # 4) ASR
    print(f"[ASR] Loading Whisper {args.asr_model} …")
    wmodel = get_whisper(args.asr_model, args.asr_device, args.asr_compute)
    words = transcribe_words_chunked(
        mix16, wmodel,
        chunk_sec=120.0, hop_sec=120.0,
        vad_params=dict(
            threshold=0.30,
            min_speech_duration_ms=150,
            min_silence_duration_ms=800,
            speech_pad_ms=120
        )
    )
    if not words:
        final_path = (Path(args.out_dir)/wav_path.stem/f"{wav_path.stem}_cleaned.txt")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_text("No speech recognized.", encoding="utf-8")
        if getattr(args, "keep_only_cleaned", False):
            cleanup_workdir(final_path.parent, keep=[final_path])
        print(f"[DONE] {wav_path.stem} (empty ASR)")
        return True
    with open(work/f"{base}_mix_words.json","w",encoding="utf-8") as f: json.dump(words, f, indent=2)

    spans = load_segments_csv(seg_csv)
    labeled = assign(words, spans)

    # 5) group into turns
    max_gap = estimate_max_gap(labeled) if args.auto_gap else args.max_gap
    turns = group_turns(labeled, max_gap=max_gap, speaker_switch_pad=args.speaker_switch_pad)
    if not turns:
        (work/f"{wav_path.stem}_mix_transcript_labeled.txt").write_text("", encoding="utf-8")
        (work/f"{wav_path.stem}_mix_transcript_labeled.jsonl").write_text("", encoding="utf-8")
        final_path = work/f"{wav_path.stem}_cleaned.txt"
        final_path.write_text("No turns found.", encoding="utf-8")
        if getattr(args, "keep_only_cleaned", False):
            cleanup_workdir(work, keep=[final_path])
        print(f"[DONE] {base} (no turns)")
        return True
    turns = fix_boundary_fragments(turns)

    # 6) write labeled transcript
    out_txt = work/f"{base}_mix_transcript_labeled.txt"
    out_jsonl = work/f"{base}_mix_transcript_labeled.jsonl"
    with open(out_txt,"w",encoding="utf-8") as f_txt, open(out_jsonl,"w",encoding="utf-8") as f_js:
        for seg in turns:
            if not seg: continue
            spk = seg[0]["speaker"]; line = punct(seg)
            f_txt.write(f"{spk}: {line}\n")
            f_js.write(json.dumps({
                "speaker": spk, "start": seg[0]["start"], "end": seg[-1]["end"], "text": line
            }, ensure_ascii=False) + "\n")

    # 7) LLM clean with timestamps
    final_path = work / f"{base}_cleaned.txt"
    if args.use_llm:
        role_lines = relabel_turns_by_content_with_times(turns)
        if not role_lines:
            final_path.write_text("No content to clean.", encoding="utf-8")
            if getattr(args, "keep_only_cleaned", False):
                cleanup_workdir(work, keep=[final_path])
            print(f"[DONE] {base} (no content)")
            return True
        role_lines = [(r, basic_cleanup_text(t), s, e) for (r,t,s,e) in role_lines]
        chunks = chunk_role_lines(role_lines, max_chars=12000, max_turns=180)
        try:
            cleaned = llm_clean_chunks_ts(
                chunks,
                model=args.openai_model,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens
            )
        except Exception as e:
            sys.stderr.write(f"LLM clean failed: {e}\n")
            cleaned = "\n\n".join(f"{r} [{fmt_ts(s)}–{fmt_ts(e)}]: {t}" for r,t,s,e in role_lines)
    else:
        role_lines = relabel_turns_by_content_with_times(turns)
        cleaned = "\n\n".join(f"{r} [{fmt_ts(s)}–{fmt_ts(e)}]: {basic_cleanup_text(t)}" for r,t,s,e in role_lines)
    with open(final_path,"w",encoding="utf-8") as f:
        f.write(cleaned)

    # optional cleanup
    if getattr(args, "keep_only_cleaned", False):
        cleanup_workdir(work, keep=[final_path])

    print(f"[DONE] {base}")
    # free some memory pressure between files
    del words, labeled, turns
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser(description="End-to-end batch: analyze→enhance→demucs→denoise→mix→diarize→ASR→label→LLM clean.")
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rnnoise-model", required=True, help=".rnnn model path for arnndn")
    # Demucs
    ap.add_argument("--demucs-device", choices=["auto","cuda","cpu"], default="auto")
    ap.add_argument("--demucs-model", default="htdemucs")
    ap.add_argument("--demucs-segment", type=int, default=10)
    ap.add_argument("--demucs-overlap", type=float, default=0.25)
    ap.add_argument("--demucs-shifts", type=int, default=2)
    # Whisper
    ap.add_argument("--asr-model", default=os.environ.get("ASR_MODEL_SIZE","large-v3"))
    ap.add_argument("--asr-device", default=os.environ.get("ASR_DEVICE","auto"))
    ap.add_argument("--asr-compute", default=os.environ.get("ASR_COMPUTE","auto"))
    # Grouping
    ap.add_argument("--max-gap", type=float, default=1.2)
    ap.add_argument("--auto-gap", action="store_true")
    ap.add_argument("--speaker-switch-pad", type=float, default=0.2)
    # LLM clean
    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument("--openai-model", default="gpt-4.1-mini")
    ap.add_argument("--interviewer-hint", default="I'm Ellen")  # kept for compatibility
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=8000)
    # Cleanup
    ap.add_argument("--keep-only-cleaned", action="store_true", help="Delete all intermediates, keep only *_cleaned.txt")
    args = ap.parse_args()

    in_dir = Path(args.in_dir); out_dir = Path(args.out_dir)
    ensure(out_dir)

    wavs = sorted([Path(p) for p in glob.glob(str(in_dir/"*.wav"))])
    if not wavs:
        print("No .wav files found."); sys.exit(1)

    for i, w in enumerate(wavs, 1):
        print(f"[FILE {i}/{len(wavs)}] {w.name}", flush=True)
        try:
            process_one(w, args)
        except Exception as e:
            sys.stderr.write(f"[ERROR] {w.name}: {e}\n")
            continue

if __name__ == "__main__":
    main()
