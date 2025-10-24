E-DAIC_Transcription

Batch pipeline to produce higher-quality, time-aligned transcripts for the Extended Distress Analysis Interview Corpus (E-DAIC) audio. The script enhances speech, isolates vocals, diarizes speakers, performs ASR with word timestamps, merges with diarization, and optionally LLM-cleans while preserving timestamps.

Dataset context

E-DAIC extends DAIC-WOZ with semi-clinical interviews conducted by the virtual interviewer “Ellie.” WoZ sessions are human-operated; other sessions are fully autonomous. Train/dev include both; test is autonomous only. Session IDs 300–492 are WoZ and 600–718 are autonomous. Each participant folder typically contains XXX_AUDIO.wav, XXX_Transcript.csv, and precomputed audio/visual features; labels include PHQ-8 and PTSD (PCL-C) fields and item-level PHQ-8 responses. 
DAIC-WOZ
+2
DAIC-WOZ
+2

What this script does

batch_asr_pipeline.py processes .wav files in bulk:

Analyze audio → adaptive knobs
Measures RMS/SNR, spectral “rumble”/“hiss,” DC offset, and derives per-file parameters for HP/LP filters, RNNoise mix, de-esser, gating, compression, and EQ.

Speech enhancement and denoising (FFmpeg)
Adaptive chain with dynaudnorm + dcshift → low-band EQ cuts → high-pass → optional RNNoise (arnndn) + afftdn → additional presence EQ, de-essing, gating, compression, and limiting.

Source separation (Demucs)
Splits to vocals.wav and no_vocals.wav; applies tailored denoise/EQ to each; remixes at 16 kHz with residual kept ~8 dB below vocals.

Speaker diarization (pyannote 3.1)
Runs pyannote/speaker-diarization-3.1 at 16 kHz and saves *_segments.csv.

ASR (faster-whisper)
Word-timestamp decoding with fallback temperatures and VAD. Chunks audio, removes repeated 6-grams, writes *_mix_words.json.

Align words to speakers
Labels each word by overlap with diarization spans; estimates max same-speaker gap; groups into turns; fixes boundary fragments.

Outputs transcripts

*_mix_transcript_labeled.txt and *_mix_transcript_labeled.jsonl (speaker turns with start/end per turn).

*_cleaned.txt: role-aware lines with preserved timestamps. If --use-llm, performs constrained LLM cleanup that keeps line count and timestamps fixed.

Optional cleanup
--keep-only-cleaned prunes intermediates.

Relation to other E-DAIC modalities

This pipeline targets the audio channel (XXX_AUDIO.wav). It can be combined downstream with E-DAIC’s distributed features for multimodal modeling (OpenSMILE eGeMAPS/MFCC BoAW, DenseNet/VGG audio spectrogram embeddings, OpenFace pose/gaze/AUs, and train/dev/test label files with PHQ-8 and PCL-C). 
DAIC-WOZ
+2
DAIC-WOZ
+2

Requirements

Python 3.10+

FFmpeg built with arnndn (optional but recommended)

RNNoise .rnnn model file for --rnnoise-model

Demucs (CLI)

PyTorch with CUDA if available

pyannote.audio pipeline pyannote/speaker-diarization-3.1 (HF auth if required)

faster-whisper

GPU is optional but speeds Demucs, pyannote, and Whisper.

Installation (minimal)
pip install numpy pandas soundfile torch faster-whisper pyannote.audio
# install demucs CLI; ensure ffmpeg is on PATH and built with arnndn

Usage
python batch_asr_pipeline.py \
  --in-dir /path/to/wavs \
  --out-dir /path/to/out \
  --rnnoise-model /path/to/quiet.rnnn \
  --demucs-device auto --demucs-model htdemucs \
  --asr-model large-v3 --asr-device auto --asr-compute auto \
  --auto-gap --speaker-switch-pad 0.2 \
  --use-llm --openai-model gpt-4.1-mini --temperature 0.0 \
  --keep-only-cleaned


Input expectation: flat directory of .wav files. The script creates one work folder per file under --out-dir.

Key outputs per file
<OUT>/<BASE>/
  *_vocal_16k.wav             # enhanced vocals
  *_residual_16k.wav          # denoised residual
  *_mix_16k.wav               # adaptive remix for diarization/ASR
  *_segments.csv              # diarization turns
  *_mix_words.json            # word-level timestamps
  *_mix_transcript_labeled.txt
  *_mix_transcript_labeled.jsonl
  *_cleaned.txt               # final timestamped dialogue