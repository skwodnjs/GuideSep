# GuideSep Minimal Inference

A minimal inference-only wrapper around [YutongWen/GuideSep](https://github.com/YutongWen/GuideSep).

This repository intentionally removes the original training, evaluation, dataset, logging, Hydra, and Lightning code paths. It keeps only a Jupyter workflow for separating a target source from a mixture using an aligned guide waveform.

## Usage

1. Create an environment and install the dependencies:

```bash
pip install -r requirements.txt
```

2. Put the input files here:

```text
wav/guide.wav
wav/mix.wav
```

3. Open `inference.ipynb` and run all cells.

4. The result is written to:

```text
output/output.wav
```

The first run downloads the official `YutongCooper/GuideSep-v1` checkpoint from Hugging Face and the inference-time model source files pinned to upstream GuideSep commit `f6bcbdee55b56909a6dc4096f8f75fb93b47c202`. Downloaded runtime source is stored under `.runtime/` and is not committed.

## Current scope

Only waveform-guide conditioning is used. Mel-spectrogram positive/negative masks are deliberately disabled. Internally, the two mask-conditioning channels are filled with the same `-1` sentinel used by the original GuideSep model when masks are dropped.

`guide.wav` and `mix.wav` should be time-aligned from the beginning. Audio is converted to mono and resampled to 16 kHz for inference.

Future extensions can add automatic masks and explicit lower/upper frequency guidance while keeping this mask-free output as the baseline.

## Upstream

GuideSep was introduced in Y. Wen, M. Kim, and P. Smaragdis, “User-Guided Generative Source Separation,” ISMIR 2025. The upstream source is MIT licensed; see `LICENSE`.
