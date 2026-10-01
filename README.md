# GuideSep Minimal Inference

Minimal inference workflow for guide-audio source separation with GuideSep.

Input convention:

```text
wav/guide.wav
wav/mix.wav
```

Notebook:

```text
inference_guidesep.ipynb
```

## Installation

PyTorch is intentionally not listed in `requirements.txt` because the correct build depends on your platform and compute backend.

For an NVIDIA GPU with CUDA 12.8, install PyTorch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

For CPU-only use:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If you need another CUDA version, use the matching command from the official PyTorch installation selector.

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

You can verify CUDA availability with:

```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

The notebook automatically uses CUDA when `torch.cuda.is_available()` is `True`; otherwise it falls back to CPU.

## GuideSep

Open `inference_guidesep.ipynb` and run all cells.

Outputs:

```text
output/target_guidesep.wav
output/residual_guidesep.wav
```

`target_guidesep.wav` is the source estimated by GuideSep. `residual_guidesep.wav` is the remainder of the normalized input mixture, computed as `mixture - target`, so the two outputs reconstruct the processed mixture over their common output length.

Running GuideSep again overwrites these same two output files.

The first run downloads the official `YutongCooper/GuideSep-v1` checkpoint from Hugging Face and the inference-time model source files pinned to upstream GuideSep commit `f6bcbdee55b56909a6dc4096f8f75fb93b47c202`. Downloaded runtime source is stored under `.runtime/` and is not committed.

GuideSep WAV input and output are handled with `soundfile`. If an input file is not already at 16 kHz, it is resampled with `scipy.signal.resample_poly`. `torchaudio` and `torchcodec` are not required.

`guide.wav` and `mix.wav` should be time-aligned from the beginning. Stereo or multichannel audio is converted to mono before GuideSep inference.

Only waveform-guide conditioning is currently used. Mel-spectrogram positive/negative masks are deliberately disabled. Internally, the two mask-conditioning channels are filled with the same `-1` sentinel used by the original GuideSep model when masks are dropped.

## Upstream

GuideSep was introduced in Y. Wen, M. Kim, and P. Smaragdis, “User-Guided Generative Source Separation,” ISMIR 2025. The upstream GuideSep source is MIT licensed.
