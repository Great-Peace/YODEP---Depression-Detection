# Installation Notes

## System Requirements

- Python 3.10 or higher
- 8 GB RAM minimum (16 GB recommended for SSL model extraction)
- ~10 GB disk space for model weights (HuBERT, WavLM, BERT)
- GPU optional but recommended for C4 SSL extraction

## Platform-Specific Instructions

### Ubuntu 22.04

```bash
# System dependencies
sudo apt-get update
sudo apt-get install -y build-essential python3-dev portaudio19-dev libsndfile1

# Python environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

For CUDA support (replace cu118 with your CUDA version):
```bash
pip install torch==2.2.1+cu118 torchaudio==2.2.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

### macOS (Apple Silicon — M1/M2/M3)

```bash
# Xcode command line tools (required for Parselmouth)
xcode-select --install

# Homebrew dependencies
brew install libsndfile

# Python environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

MPS (Metal Performance Shaders) is detected automatically for GPU acceleration.

### Windows 11

1. Install [Python 3.10+](https://www.python.org/downloads/)
2. Install [Build Tools for Visual Studio 2022](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) (required for Parselmouth)
   - In the installer, select: "C++ build tools" workload
3. In PowerShell or Command Prompt:
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

## Verifying Installation

```bash
python -c "import parselmouth; print('Parselmouth OK')"
python -c "import opensmile; print('openSMILE OK')"
python -c "import torch; print(f'PyTorch OK, CUDA={torch.cuda.is_available()}')"
python -c "import transformers; print('transformers OK')"
python scripts/verify_yodep.py --help
```

## Parselmouth Build Errors

If `pip install praat-parselmouth` fails with compiler errors:

**Ubuntu — missing pybind11:**
```bash
pip install pybind11
pip install praat-parselmouth --no-binary praat-parselmouth
```

**macOS — linking error:**
```bash
export MACOSX_DEPLOYMENT_TARGET=10.15
pip install praat-parselmouth
```

**All platforms — pre-built wheel:**
Check https://github.com/YannickJadoul/Parselmouth/releases for a pre-built
wheel matching your Python version.

## openSMILE Version Compatibility

openSMILE 2.5.0 is a pure Python package that bundles the C++ binaries.
If you encounter import issues:
```bash
pip uninstall opensmile
pip install opensmile==2.5.0 --no-cache-dir
```

## HuggingFace Model Download

First run of C4 will download ~360 MB (HuBERT) and ~360 MB (WavLM).
To pre-download:
```python
from transformers import AutoModel, AutoFeatureExtractor
AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
AutoModel.from_pretrained("facebook/hubert-base-ls960")
AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
AutoModel.from_pretrained("microsoft/wavlm-base")
```

Set HF_HOME to control cache location:
```bash
export HF_HOME=/path/with/enough/space/.cache/huggingface
```
