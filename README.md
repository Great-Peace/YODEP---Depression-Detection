# YODEP: Yoruba-English Acted Depression Speech Corpus and F0 Transferability Study

**Thesis:** *Can F0 Be Trusted? Investigating Fundamental Frequency as a Depression Biomarker Across English and Yoruba*
**Supervisor:** Prof. Prasenjit Mitra, Carnegie Mellon University Africa
**Target venue:** InterSpeech 2026 / IEEE ICASSP 2027

---

## Overview

YODEP is a novel bilingual acted speech corpus and experimental pipeline investigating whether fundamental frequency (F0) — the primary biomarker used in every competitive speech-based depression detection system — behaves differently in English and Yoruba, a West African lexical tone language where F0 encodes word meaning.

**Core research question:** Does removing F0 features from a depression classifier improve cross-lingual transferability from English to Yoruba? If yes, this provides the first empirical evidence that F0-based depression models are undeployable in African tonal language contexts.

---

## Experiment Conditions

```
┌─────────────────────────────────────────────────────────────────────────┐
│  C1  Full Features        Prosodic (F0) + Spectral (MFCCs) + Glottal   │
│                           Standard approach — reproduces prior work     │
├─────────────────────────────────────────────────────────────────────────┤
│  C2  F0-Ablated           Spectral (MFCCs) + Glottal only              │
│                           F0 strictly excluded (assertion enforced)     │
├─────────────────────────────────────────────────────────────────────────┤
│  C3  Glottal Only         Jitter + Shimmer + HNR + CPPS only           │
│                           Language-agnostic lower bound                 │
├─────────────────────────────────────────────────────────────────────────┤
│  C4  SSL Embeddings       HuBERT-base / WavLM-base mean-pooled (768d)  │
│                           Foundation model baseline                     │
└─────────────────────────────────────────────────────────────────────────┘

Multimodal extensions: C1-MM, C2-MM, C3-MM, C4-MM (audio + BERT [CLS])
Transfer experiments:  C1-XFER, C2-XFER, C3-XFER (EN→YO zero-shot)
```

---

## Repository Structure

```
yodep/
├── config/           config.yaml, experiment_config.yaml
├── data/
│   ├── sentences.json          Fixed sentence stimuli (EN + YO)
│   ├── daic_woz/               Place DAIC-WOZ files here (raw/)
│   └── yodep/                  Place YODEP .wav files here (raw/)
├── src/
│   ├── data/                   Loaders and audio utilities
│   ├── features/               Prosodic, spectral, glottal, SSL, text
│   ├── models/                 SVM, LR, RF, MLP
│   ├── evaluation/             Metrics, LOSO, Wilcoxon, actor quality
│   ├── visualisation/          F0 contours, feature importance, tables
│   └── utils/                  Logging, seeding, caching
├── experiments/      run_daic_validation.py, run_yodep_main.py, run_transfer.py
├── scripts/          verify_yodep.py, extract_all_features.py, generate_metadata_template.py
├── tests/            pytest test suite
├── notebooks/        Exploratory analysis notebooks
└── results/          Auto-generated tables (CSV + LaTeX) and figures (PNG)
```

---

## Installation

### 1. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**PyTorch with CUDA** (replace `cu118` with your CUDA version):
```bash
pip install torch==2.2.1+cu118 torchaudio==2.2.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

**PyTorch on Apple Silicon** (MPS):
```bash
pip install torch==2.2.1 torchaudio==2.2.1
```

### 3. Install the package

```bash
pip install -e .
```

---

## Dataset Setup

### YODEP (primary dataset)

1. Place all YODEP `.wav` files in `data/yodep/raw/`
2. Files **must** follow the naming convention exactly:
   ```
   [SpeakerID]_[Language]_[Condition]_[Sentence]_[Take].wav
   ```
   Examples:
   ```
   P01_EN_NORMAL_S1_T1.wav
   P01_YO_DEPRESSED_S3_T1.wav
   P07_EN_NORMAL_S5_T2.wav
   ```
3. Generate the metadata template:
   ```bash
   python scripts/generate_metadata_template.py
   ```
4. Fill in `data/yodep/metadata.csv` with one row per speaker.

### DAIC-WOZ (validation only)

DAIC-WOZ requires signing a Data Use Agreement at https://dcapswoz.ict.usc.edu/

Once access is granted:
1. Place the extracted files under `data/daic_woz/raw/` with the structure:
   ```
   data/daic_woz/raw/
   ├── labels.csv          (or phq8_labels.csv)
   ├── train_split.csv
   ├── dev_split.csv
   └── 300/
       ├── 300_AUDIO.wav
       └── 300_TRANSCRIPT.csv
   ```

---

## Running the Pipeline

### Step 1: Verify YODEP structure

```bash
python scripts/verify_yodep.py
python scripts/verify_yodep.py --verbose     # print each file
python scripts/verify_yodep.py --help
```

### Step 2: Extract all features (optional — will be done automatically)

```bash
python scripts/extract_all_features.py
python scripts/extract_all_features.py --dataset yodep --conditions C1 C2 C3
```

### Step 3: Run individual experiments

```bash
# DAIC-WOZ pipeline validation
python experiments/run_daic_validation.py

# YODEP main experiment (all conditions, LOSO CV)
python experiments/run_yodep_main.py
python experiments/run_yodep_main.py --language EN
python experiments/run_yodep_main.py --conditions C1 C2 C3

# Cross-lingual transfer
python experiments/run_transfer.py
```

### Step 4: Run everything

```bash
python experiments/run_all.py
python experiments/run_all.py --skip-daic    # skip DAIC-WOZ if unavailable
```

---

## Docker

### Build the image

```bash
docker build -t yodep:latest .
```

### Run experiments

```bash
# Verify YODEP files
docker compose run --rm yodep-verify

# Pre-extract features (recommended before long runs)
docker compose run --rm yodep-extract

# Main experiments (LOSO CV, all conditions)
docker compose up yodep-main

# Everything at once
docker compose up yodep-all
```

### Data mounting

Edit `docker-compose.yml` to point volume mounts at your actual data paths:

```yaml
volumes:
  - /path/to/your/yodep/wavs:/app/data/yodep/raw:ro
  - /path/to/results:/app/results
```

### GPU

The compose file requests 1 GPU (`count: 1`).  All sub-1B models (HuBERT, WavLM, BERT) fit comfortably in a single 16 GB GPU.  A 40 GB GPU has ample headroom.

### Checkpoint/resume

If the job is interrupted, restart with the same command — completed LOSO folds and MLP epoch states are checkpointed to the `yodep-exp-checkpoints` and `yodep-feature-cache` Docker volumes and resume automatically.

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/test_pipeline_assertions.py -v    # C2 assertion tests specifically
pytest tests/ --cov=src --cov-report=term
```

---

## Output Files

All outputs are auto-saved. No manual saving required.

| Path | Contents |
|------|----------|
| `results/tables/table1_daic_validation.csv` + `.tex` | DAIC-WOZ validation (Table 1) |
| `results/tables/table2_yodep_main.csv` + `.tex` | YODEP main results (Table 2) |
| `results/tables/table3_transfer.csv` + `.tex` | Cross-lingual transfer (Table 3) |
| `results/tables/table4_actor_quality.csv` + `.tex` | Actor quality ranked table (Table 4) |
| `results/tables/table5_wilcoxon.csv` + `.tex` | Wilcoxon summary (Table 5) |
| `results/figures/f0_contour_{speaker}_{lang}.png` | Per-speaker F0 contours |
| `results/figures/f0_contour_summary.png` | Summary figure (key paper figure) |
| `results/figures/feature_importance_C1_*_RF.png` | RF feature importance |
| `results/figures/transfer_curves_C1_C2_C3.png` | Transfer performance chart |
| `logs/` | Timestamped experiment logs |

---

## Citation

```bibtex
@dataset{yodep2025,
  title     = {{YODEP}: Yoruba-English Acted Depression Speech Corpus},
  author    = {[Author Name]},
  year      = {2025},
  institution = {Carnegie Mellon University Africa},
  note      = {Master's thesis dataset. Supervisor: Prof. Prasenjit Mitra.},
  license   = {CC BY 4.0}
}

@inproceedings{yodep_interspeech2026,
  title     = {Can {F0} Be Trusted? Investigating Fundamental Frequency
               as a Depression Biomarker Across English and Yoruba},
  author    = {[Author Name] and Prasenjit Mitra},
  booktitle = {Proc. Interspeech 2026},
  year      = {2026},
  note      = {[to appear]}
}
```

---

## Licence

- **Corpus** (`data/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Code** (`src/`, `experiments/`, `scripts/`, `tests/`): MIT

See [LICENSE](LICENSE) for full terms.

---

## Acknowledgements

This research was supported by Carnegie Mellon University Africa.
Supervisor: Prof. Prasenjit Mitra.
Participants: [to be acknowledged after IRB approval].
Native Yoruba speaker review of sentence stimuli: [name, affiliation].

---

## Troubleshooting

### Parselmouth installation fails

**Ubuntu/Debian:**
```bash
sudo apt-get install build-essential python3-dev
pip install praat-parselmouth
```

**macOS:**
```bash
xcode-select --install
pip install praat-parselmouth
```

**Windows:** Install [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022), then:
```bash
pip install praat-parselmouth
```

### openSMILE issues

openSMILE is installed as a pure Python package (`opensmile`). No separate binary is needed. If you see import errors:
```bash
pip install opensmile==2.5.0
```

### CUDA not detected

```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

If False, reinstall PyTorch with the correct CUDA version for your GPU driver.

### DAIC-WOZ access denied

The DAIC-WOZ corpus is restricted. Apply at https://dcapswoz.ict.usc.edu/
DAIC-WOZ is used only for pipeline validation (Table 1). All main results use YODEP only. You can run all YODEP experiments with `--skip-daic`.

### HuggingFace models not downloading

Set the cache directory if the default is on a small partition:
```bash
export HF_HOME=/path/to/large/disk/.cache/huggingface
```

Or in Python before importing transformers:
```python
import os
os.environ["HF_HOME"] = "/path/to/large/disk/.cache/huggingface"
```
