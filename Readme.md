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

If you want GPU support, make sure to install a PyTorch version,
that is compatible with your CUDA installation.

Install all required packages using:

```bash
pip install -r EMCode/requirements.txt
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

## 4. Counterfactual explainer

If you experience errors when running the counterfactual attack script
(e.g. TensorFlow GPU errors, libdevice missing, or NLTK lookup failures), follow the steps below.

### 1. Download required NLTK resources

TextAttack depends on several NLTK corpora and tokenizers.  
If you get errors related to `punkt`, `wordnet`, or POS tagging, run the following once:

```bash
python - <<EOF
import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
EOF
```

### 2. Fix TensorFlow GPU error: libdevice not found

If you see an error like:

```bash
JIT compilation failed
libdevice not found at ./libdevice.10.bc
```

Verify its existence like this:

```bash
$ find / -type d -name nvvm 2>/dev/null
/usr/lib/cuda/nvvm
$ cd /usr/lib/cuda/nvvm
/usr/lib/cuda/nvvm$ ls
libdevice
/usr/lib/cuda/nvvm$ cd libdevice
/usr/lib/cuda/nvvm/libdevice$ ls
libdevice.10.bc
```

#### Set the required environment variable

from the console:
```bash
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
```

or in the IDE in Environment variables for the run Profile
```ini
XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
```

### Notes

- CUDA Toolkit must be installed (not only NVIDIA drivers).
- If GPU setup fails, the attack will still run on CPU but much slower.
- This setup was tested with CUDA installed under /usr/lib/cuda.