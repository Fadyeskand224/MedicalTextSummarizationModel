# 🧠 Medical Text Summarization Model
> Abstractive summarization for biomedical literature and systematic reviews using a Transformer/GPT-style decoder trained on PubMed & MS² data.

<p align="left">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue.svg"></a>
  <a href="#"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="#"><img alt="Git LFS" src="https://img.shields.io/badge/Large%20Files-Git%20LFS-critical.svg"></a>
  <a href="#"><img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
</p>

---

## 📚 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Repository Structure](#-repository-structure)
- [Quickstart](#-quickstart)
- [Setup & Installation](#-setup--installation)
- [Training](#-training)
- [Evaluation & Metrics](#-evaluation--metrics)
- [Inference (Generate Summaries)](#-inference-generate-summaries)
- [Datasets](#-datasets)
- [Model Checkpoints](#-model-checkpoints)
- [Configuration & Reproducibility](#-configuration--reproducibility)
- [Troubleshooting](#-troubleshooting)
- [FAQ](#-faq)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)
- [Citation](#-citation)
- [Contact](#-contact)

---

## 🧾 Overview
This repository delivers an end-to-end workflow for **abstractive summarization of medical texts**: data prep → tokenization → model training → evaluation → inference. It’s geared toward **biomedical abstracts, clinical trials, and systematic reviews**, producing concise summaries that retain critical findings and outcomes.

---

## ✨ Features
- ⚙️ **Custom GPT-style decoder** implemented with PyTorch.
- 🩺 **Domain-specific corpora**: PubMed & MS² medical review data.
- 🔁 **Complete pipeline**: preprocessing → training → evaluation → inference (notebook driven).
- 📊 **Metrics**: ROUGE-1/2/L and BLEU (with hooks for Precision/Recall/F1).
- 💾 **Reproducible artifacts**: versioned checkpoints (`gpt_decoder_epoch*.pt`) via **Git LFS**.
- 📓 **Interactive experimentation** in `text_summ_model_training.ipynb`.
- 🧑‍⚕️ **Healthcare-aware preprocessing** (e.g., de-noising references/section headers).

---

## 🗂️ Repository Structure
```plaintext
MedicalTextSummarizationModel/
│
├── Dataset.zip                               # Original dataset bundle (use Git LFS)
├── text_summ_model_training.ipynb            # Main training & evaluation notebook
├── gpt_decoder_epoch1.pt                     # Saved model checkpoint (LFS)
├── gpt_decoder_epoch2.pt                     # Saved model checkpoint (LFS)
├── gpt_decoder_epoch3.pt                     # Saved model checkpoint (LFS; best)
├── Medical_Txt_Summarization_Model_Documentation.pdf
├── requirements.txt                          # Python dependencies
└── readme.txt                                # Legacy notes
```

> 🔴 **Large files are tracked with Git LFS.** If you clone without LFS, checkpoints/datasets will appear as small pointer files.

---

## ⚡ Quickstart
```bash
# 1) Clone
git clone https://github.com/Fadyeskand224/MedicalTextSummarizationModel.git
cd MedicalTextSummarizationModel

# 2) Enable Git LFS (important for checkpoints/dataset)
git lfs install
git lfs pull

# 3) Create env & install deps
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 4) Unpack dataset bundle (if present)
unzip Dataset.zip -d Dataset/

# 5) Open the notebook and run the pipeline
jupyter notebook text_summ_model_training.ipynb
```

---

## 🧪 Training
Open the notebook and execute cells in order:
1. **Config**: paths, hyperparameters, random seeds.
2. **Data Prep**: load & clean texts; split train/val/test.
3. **Tokenization**: (e.g., WordPiece/BPE) with truncation strategies.
4. **Model**: initialize GPT-style decoder; set optimizer/scheduler.
5. **Train Loop**: run for N epochs; save checkpoints per epoch.
6. **Eval**: compute ROUGE/BLEU; pick best checkpoint.

---

## 📈 Evaluation & Metrics
| Model Checkpoint       | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|------------------------|:------:|:------:|:------:|:----:|
| gpt_decoder_epoch1.pt  |  XX.X  |  XX.X  |  XX.X  | XX.X |
| gpt_decoder_epoch2.pt  |  XX.X  |  XX.X  |  XX.X  | XX.X |
| gpt_decoder_epoch3.pt  | **XX.X**| **XX.X**| **XX.X**|**XX.X**|

---

## 🧾 Inference (Generate Summaries)
```python
import torch
from your_model_impl import YourSummarizerModel, YourTokenizer

ckpt_path = "gpt_decoder_epoch3.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = YourTokenizer.load_pretrained_or_local(...)
model = YourSummarizerModel(...)
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state)
model.to(device).eval()

article = """OBJECTIVE: Evaluate the efficacy..."""
inputs = tokenizer(article, return_tensors="pt", truncation=True, max_length=2048).to(device)

with torch.no_grad():
    summary_ids = model.generate(
        **inputs,
        max_new_tokens=220,
        temperature=0.7,
        top_p=0.9,
        do_sample=False
    )

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("SUMMARY:\n", summary)
```

---

## 🗄️ Datasets
- **PubMed Summarization**
- **MS² Medical Systematic Reviews**

⚠️ Use datasets for **research/educational purposes** only.  

---

## 💾 Model Checkpoints
- `gpt_decoder_epoch1.pt` — early baseline
- `gpt_decoder_epoch2.pt` — improved mid-training
- `gpt_decoder_epoch3.pt` — final/best

---

## 🧩 Configuration & Reproducibility
```yaml
model:
  d_model: 768
  n_layers: 12
  n_heads: 12
  dropout: 0.1
train:
  batch_size: 8
  max_input_tokens: 2048
  max_summary_tokens: 256
  lr: 3e-5
  weight_decay: 0.01
  warmup_steps: 1000
  epochs: 3
```

---

## 🧯 Troubleshooting
- **Large files missing:** `git lfs install && git lfs pull`
- **CUDA OOM:** reduce batch size or max tokens
- **Slow training:** try GPU or smaller configs

---

## ❓ FAQ
**Q:** Can I fine-tune on custom data?  
**A:** Yes, just point the notebook to your dataset.  

**Q:** Can I deploy as an API?  
**A:** Wrap inference in FastAPI/Flask.  

---

## 🗺️ Roadmap
- Add **Pegasus**/**BERTSUM** baselines  
- FastAPI demo service  
- Publish pip package  

---

## 🤝 Contributing
Contributions welcome! Fork → Branch → PR.  

---

## 📜 License
MIT License. See `LICENSE`.  

---

## 🙏 Acknowledgements
- PubMed Dataset  
- MS² Dataset  
- PyTorch, Hugging Face  

---

## 📝 Citation
```bibtex
@software{medical_text_summarization_model_2025,
  author  = {Eskandr, Fady},
  title   = {Medical Text Summarization Model},
  year    = {2025},
  url     = {https://github.com/Fadyeskand224/MedicalTextSummarizationModel}
}
```

---

## 📬 Contact
- **Author:** Fady Eskandr  
- **GitHub:** https://github.com/Fadyeskand224  
- **LinkedIn:** https://www.linkedin.com/in/fady-eskandr  
