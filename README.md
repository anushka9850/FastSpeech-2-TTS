# FastSpeech 2 TTS — English (Svarah) and Telugu (IndicTTS)

This project implements a FastSpeech 2 text-to-speech pipeline for English and Telugu using the reference
implementation from [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) and the paper
[FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558).

This is a command-line project only. There is no dashboard or web UI required. The full workflow is run from the terminal using Python scripts.

The repository includes:
- separate English and Telugu preprocessing, training, and inference code
- language-specific configuration files
- checkpoint handling
- loss plotting, inference output generation, variance-adaptor plots
- technical report generation support

```
project/
├── English/                   preprocess.py  train.py  inference.py  config.yaml  checkpoints/
├── Telugu/                    preprocess.py  train.py  inference.py  config.yaml  checkpoints/
├── fs2/                      shared FastSpeech 2 package (model, aligner, front-ends, pipeline)
│   ├── transformer/
│   ├── hifigan/
│   ├── model/
│   └── text/
├── Results/
│   ├── English/
│   └── Telugu/
├── Report/
├── scripts/
├── test_sentences/
├── docs/
├── README.md
├── requirements.txt
└── .gitignore
```

## 1. Project goals

This assignment requires:
- training one FastSpeech 2 model for English using Svarah
- training one FastSpeech 2 model for Telugu using IndicTTS Telugu
- keeping train/validation data separate without overlap
- generating speech samples for the required test sentences
- plotting duration, pitch (F0), and energy predictions from the variance adaptor
- producing a technical report explaining the implementation and results

## 2. Setup

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux/macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ('averaged_perceptron_tagger_eng','averaged_perceptron_tagger','cmudict')]"
```

If you are using Hugging Face datasets, also authenticate:

```bash
huggingface-cli login
```

## 3. Quick validation

This project includes a smoke test that verifies the end-to-end pipeline without requiring full real-world dataset training.

```bash
python scripts/smoke_test.py
```

This is the fastest way to confirm that preprocessing, training, inference, and output generation all work in the current environment.

## 4. Data preparation and preprocessing

| Language | Dataset | Command |
|---|---|---|
| English | Svarah | `python English/preprocess.py` |
| Telugu | IndicTTS Telugu | `python Telugu/preprocess.py` |

The preprocessing pipeline:
- loads the dataset
- filters invalid or unsuitable utterances
- performs the language-specific text normalization
- splits data into train and validation sets without overlap
- extracts robust features for training
- writes processed data for model training

## 5. Training

```bash
python English/train.py
python Telugu/train.py
```

Optional:

```bash
python English/train.py --resume
python Telugu/train.py --resume
```

Training outputs include:
- checkpoints under each language folder
- validation and training loss logs
- loss plots saved under `Results/<Language>/`

## 6. Inference and variance-adaptor analysis

```bash
python English/inference.py
python Telugu/inference.py
```

This generates:
- speech files under `Results/<Language>/test_generated_audios/`
- duration, pitch, and energy plots under `Results/<Language>/variance_adaptor_plots/`
- per-sentence metadata values for analysis

## 7. Report generation

After generating the required outputs, create the technical report:

```bash
pip install playwright
playwright install chromium
python scripts/make_report.py
```

The report is written to:
- `Report/Technical_Report.pdf`

## 8. Results folder structure

```text
Results/
├── English/
│   ├── training_loss.png
│   ├── validation_loss.png
│   ├── test_generated_audios/
│   └── variance_adaptor_plots/
│       ├── duration/
│       ├── pitch/
│       └── energy/
└── Telugu/
    ├── training_loss.png
    ├── validation_loss.png
    ├── test_generated_audios/
    └── variance_adaptor_plots/
        ├── duration/
        ├── pitch/
        └── energy/
```

## 9. Notes for assignment completion

To fully satisfy the assignment deliverables from the PDF:
- use the real Svarah and IndicTTS Telugu datasets
- run the actual preprocessing and training commands for both languages
- save all checkpoints, generated audio, loss plots, and variance-adaptor plots
- keep the repository structure aligned with the project specification
- upload the trained checkpoints or provide accessible external links if GitHub file-size limits apply

## 10. License and acknowledgements

This project builds on the FastSpeech 2 reference implementation and other open-source components.

References:
- FastSpeech 2 paper
- ming024/FastSpeech2 repository
- HiFi-GAN reference implementation
- Noto Sans Telugu font used in the report

No dashboard is required for this project; it is designed to run from the command line as a Python TTS pipeline.
