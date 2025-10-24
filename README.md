# E-DAIC_Transcription

Batch pipeline for higher-quality, time-aligned transcripts of E-DAIC audio. It enhances speech, isolates vocals, diarizes speakers, runs ASR with word timestamps, aligns words to diarization, and optionally performs constrained LLM cleanup while preserving timestamps.

## Dataset context

E-DAIC extends DAIC-WOZ with semi-clinical interviews by the virtual interviewer “Ellie.” Wizard-of-Oz (WoZ) sessions are human-operated; others are autonomous. Train and dev include both; test is autonomous only. Typical per-participant files include `XXX_AUDIO.wav`, `XXX_Transcript.csv`, precomputed audio/visual features, and labels such as PHQ-8 and PTSD (PCL-C) with item-level PHQ-8 responses.

## Pipeline

1. **Analyze audio → adaptive knobs**  
   Computes RMS and SNR, rumble and hiss, DC offset. Derives per-file HP and LP, RNNoise mix, de-esser, gate, compression, and EQ.

2. **Speech enhancement and denoising (FFmpeg)**  
   `dynaudnorm` + `dcshift` → low-band EQ cuts → high-pass → optional RNNoise (`arnndn`) + `afftdn` → presence EQ, de-essing, gating, compression, limiting.

3. **Source separation (Demucs)**  
   Splits to `vocals.wav` and `no_vocals.wav`. Applies tailored denoise and EQ per stem. Remixes at 16 kHz with residual about 8 dB below vocals.

4. **Speaker diarization (pyannote 3.1)**  
   Runs `pyannote/speaker-diarization-3.1` and writes `*_segments.csv`.

5. **ASR (faster-whisper)**  
   Word-timestamp decoding with VAD and fallback temperatures. Chunks audio, removes repeated 6-grams, writes `*_mix_words.json`.

6. **Align words to speakers**  
   Labels words by overlap with diarization spans. Estimates max same-speaker gap. Groups into turns. Fixes boundary fragments.

7. **Outputs transcripts**  
   - `*_mix_transcript_labeled.txt` and `.jsonl` with turn start and end.  
   - `*_cleaned.txt` with role-aware lines and preserved timestamps. With `--use-llm`, applies constrained cleanup that keeps line count and timestamps fixed.

8. **Optional cleanup**  
   `--keep-only-cleaned` removes intermediates.

## Relation to other E-DAIC modalities

Targets the audio channel (`XXX_AUDIO.wav`). Downstream you can join with distributed features for multimodal work (OpenSMILE eGeMAPS and MFCC BoAW, DenseNet or VGG spectrogram embeddings, OpenFace pose, gaze, and AUs) and label files (PHQ-8, PCL-C).

## Requirements

- Python 3.10+
- FFmpeg (with `arnndn` recommended)
- RNNoise `.rnnn` model for `--rnnoise-model`
- Demucs CLI
- PyTorch (CUDA optional)
- `pyannote.audio` with `pyannote/speaker-diarization-3.1`
- `faster-whisper`

GPU is optional but speeds Demucs, pyannote, and Whisper.

## Installation

```bash
pip install numpy pandas soundfile torch faster-whisper pyannote.audio
# install demucs CLI; ensure ffmpeg is on PATH and supports arnndn
```

## Usage

```bash
python batch_asr_pipeline.py   --in-dir /path/to/wavs   --out-dir /path/to/out   --rnnoise-model /path/to/quiet.rnnn   --demucs-device auto --demucs-model htdemucs   --asr-model large-v3 --asr-device auto --asr-compute auto   --auto-gap --speaker-switch-pad 0.2   --use-llm --openai-model gpt-4.1-mini --temperature 0.0   --keep-only-cleaned
```

> Input: flat directory of `.wav` files. One work folder per file is created under `--out-dir`.

## Key outputs per file

```
<OUT>/<BASE>/
  *_vocal_16k.wav               # enhanced vocals
  *_residual_16k.wav            # denoised residual
  *_mix_16k.wav                 # remix for diarization/ASR
  *_segments.csv                # diarization turns
  *_mix_words.json              # word-level timestamps
  *_mix_transcript_labeled.txt
  *_mix_transcript_labeled.jsonl
  *_cleaned.txt                 # final timestamped dialogue
```

## References

- Gratch J, Artstein R, Lucas GM, Stratou G, Scherer S, Nazarian A, Wood R, Boberg J, DeVault D, Marsella S, Traum DR. The Distress Analysis Interview Corpus of Human and Computer Interviews. In: LREC 2014. pp. 3123-3128.
- DeVault D, Artstein R, Benn G, Dey T, Fast E, Gainer A, Georgila K, Gratch J, Hartholt A, Lhommet M, Lucas G, Marsella S, Morbini F, Nazarian A, Scherer S, Stratou G, Suri A, Traum D, Wood R, Xu Y, Rizzo A, Morency L-P. SimSensei kiosk: A virtual human interviewer for healthcare decision support. In: AAMAS 2014.
- Ringeval F, Schuller B, Valstar M, Cummins N, Cowie R, Tavabi L, Schmitt M, et al. AVEC 2019: State-of-mind, depression detection, and cross-cultural affect recognition. In: AVEC 2019. pp. 3-12. ACM.
