## 1. Environment Setup

It is recommended to use a **virtual environment** to avoid dependency conflicts.

### Option A: venv (standard Python)

```bash
python3 -m venv venv
source venv/bin/activate   # Linux / macOS
# venv\Scripts\activate    # Windows
```

### Option B: conda
```bash
conda create -n xai-em python=3.10
conda activate xai-em
```

Any isolated Python environment is sufficient.

## 2. Install Dependencies
Typical dependencies include:

- torch
- transformers
- pandas
- numpy

If you want GPU support, make sure to install a  PyTorch version,
that is compatible with your CUDA installation.

Install all required packages using:

```bash
pip install -r EmCode/requirements.txt
```
or manually install the most important packages via pip:

```bash
pip install -q transformers captum pandas
```

## 3. Trained Model

The project assumes a pretrained black-box model is already available at:

```bash
models/em_bert_model.pt
```

The model uses BERT (bert-base-uncased) encodes entity pairs into a 
shared embedding space outputs match / non-match predictions

No retraining is required for explainability.