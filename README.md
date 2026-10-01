# GuideSep Minimal Inference

Minimal inference workflows for query/guide-audio source separation.

The repository currently provides two independent notebooks that use the same input convention:

```text
wav/guide.wav
wav/mix.wav
```

- `inference.ipynb` — GuideSep
- `inference_banquet.ipynb` — Banquet

## Installation

PyTorch is intentionally not listed in the requirements files because the correct build depends on your platform and compute backend.

### GuideSep

For an NVIDIA GPU with CUDA 12.8, install PyTorch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

For CPU-only use:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install the base dependencies:

```bash
pip install -r requirements.txt
```

### Banquet

Banquet internally uses `torchaudio` transforms and PaSST. Install `torch`, `torchaudio`, and `torchvision` from the same PyTorch build index. For CUDA 12.8:

```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
```

For CPU-only use:

```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then install both the base and Banquet-specific dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-banquet.txt
```

If you need another CUDA version, use the matching commands from the official PyTorch installation selector.

You can verify CUDA availability with:

```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Both notebooks automatically use CUDA when `torch.cuda.is_available()` is `True`; otherwise they fall back to CPU.

## GuideSep

Open `inference.ipynb` and run all cells.

Outputs:

```text
output/target.wav
output/residual.wav
```

`target.wav` is the source estimated by GuideSep. `residual.wav` is the remainder of the normalized input mixture, computed as `mixture - target`, so the two outputs reconstruct the processed mixture over their common output length.

The first run downloads the official `YutongCooper/GuideSep-v1` checkpoint from Hugging Face and the inference-time model source files pinned to upstream GuideSep commit `f6bcbdee55b56909a6dc4096f8f75fb93b47c202`. Downloaded runtime source is stored under `.runtime/` and is not committed.

GuideSep WAV input and output are handled with `soundfile`. If an input file is not already at 16 kHz, it is resampled with `scipy.signal.resample_poly`. `torchaudio` and `torchcodec` are not required for this notebook.

`guide.wav` and `mix.wav` should be time-aligned from the beginning. Stereo or multichannel audio is converted to mono before GuideSep inference.

Only waveform-guide conditioning is currently used. Mel-spectrogram positive/negative masks are deliberately disabled. Internally, the two mask-conditioning channels are filled with the same `-1` sentinel used by the original GuideSep model when masks are dropped.

## Banquet

Open `inference_banquet.ipynb` and run all cells.

Outputs:

```text
output/banquet_target.wav
output/banquet_residual.wav
```

Banquet is a query-based music source separation model. The notebook uses `wav/guide.wav` as the query audio and `wav/mix.wav` as the mixture. Unlike GuideSep, the guide does not need to be time-aligned with the mixture. The published inference path uses a 10-second query; shorter guides are tiled and longer guides are truncated by the upstream implementation.

The first run downloads:

- Banquet source pinned to upstream commit `79ed5bb75e5c3a40cd319d9d990cee913fc65c26` into `.runtime/banquet/`
- the official recommended `ev-pre-aug.ckpt` checkpoint from Zenodo into `.runtime/banquet/checkpoints/`
- PaSST weights as required by `hear21passt`

The Banquet model operates internally at 44.1 kHz and uses a stereo model configuration. Mono mixtures are duplicated to stereo for inference and converted back to mono for the final outputs.

Banquet itself requires `torchaudio` for model transforms and PaSST resampling. The notebook replaces only `torchaudio.load` and `torchaudio.save` with `soundfile`-based I/O, so `torchcodec` is not required.

The default Banquet inference batch size in the notebook is `4`. If CUDA runs out of memory, reduce `BATCH_SIZE` to `2` or `1`.

Both Banquet outputs are saved as floating-point WAV files, with:

```text
banquet_residual = original mixture - banquet_target
```

so target and residual reconstruct the original mixture after output-length/channel alignment.

## Upstream

GuideSep was introduced in Y. Wen, M. Kim, and P. Smaragdis, “User-Guided Generative Source Separation,” ISMIR 2025. The upstream GuideSep source is MIT licensed.

Banquet was introduced in K. N. Watcharasupat and A. Lerch, “A Stem-Agnostic Single-Decoder System for Music Source Separation Beyond Four Stems,” ISMIR 2024. The upstream Banquet source is MIT licensed. Its official model weights are published separately on Zenodo.
