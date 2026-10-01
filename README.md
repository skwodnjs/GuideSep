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

Open `inference_guidesep.ipynb`, choose the inference mode near the top of the notebook, and run all cells.

```python
MODE = "waveform"  # "waveform" or "pseudo-mask"
```

The default is `waveform`. Only the selected mode is executed.

### `waveform`

Uses the aligned `guide.wav` waveform/spectrogram as conditioning and disables both mel-mask channels with the upstream `-1` sentinel.

Outputs:

```text
output/target_guidesep_waveform.wav
output/residual_guidesep_waveform.wav
```

### `pseudo-mask`

Uses the same waveform guide plus GuideSep's upstream pseudo-mask construction. The guide mel spectrogram is thresholded and Gaussian-smoothed to form the positive pseudo mask, while the mixture mel spectrogram forms the negative pseudo mask. Both 80-bin mel masks are passed through the learned mask MLP stored in the official checkpoint before being used as conditioning channels.

The pseudo masks are not multiplied onto the generated output spectrum, so this is not a hard frequency mask.

Outputs:

```text
output/target_guidesep_pseudo-mask.wav
output/residual_guidesep_pseudo-mask.wav
```

For either mode, the residual is computed from the normalized processed mixture as:

```text
residual = mixture - target
```

The first run downloads the official `YutongCooper/GuideSep-v1` checkpoint from Hugging Face and the inference-time model source files pinned to upstream GuideSep commit `f6bcbdee55b56909a6dc4096f8f75fb93b47c202`. Downloaded runtime source is stored under `.runtime/` and is not committed.

GuideSep WAV input and output are handled with `soundfile`. If an input file is not already at 16 kHz, it is resampled with `scipy.signal.resample_poly`. The `pseudo-mask` path reproduces the default HTK mel filterbank behavior used by upstream `torchaudio.transforms.MelScale` directly in PyTorch, so `torchaudio` and `torchcodec` are not required.

`guide.wav` and `mix.wav` should be time-aligned from the beginning. Stereo or multichannel audio is converted to mono before GuideSep inference.

## Upstream

GuideSep was introduced in Y. Wen, M. Kim, and P. Smaragdis, “User-Guided Generative Source Separation,” ISMIR 2025. The upstream GuideSep source is MIT licensed.
