============================================================
   MEDICAL TEXT SUMMARIZATION MODEL
============================================================

Abstractive summarization for biomedical literature and
systematic reviews using a Transformer/GPT-style decoder
trained on PubMed & MS² data.

------------------------------------------------------------
TABLE OF CONTENTS
------------------------------------------------------------
1. Overview
2. Features
3. Repository Structure
4. Quickstart
5. Setup & Installation
6. Training
7. Evaluation & Metrics
8. Inference (Generate Summaries)
9. Datasets
10. Model Checkpoints
11. Configuration & Reproducibility
12. Troubleshooting
13. FAQ
14. Roadmap
15. Contributing
16. License
17. Acknowledgements
18. Citation
19. Contact

------------------------------------------------------------
1. OVERVIEW
------------------------------------------------------------
This repository delivers an end-to-end workflow for
abstractive summarization of medical texts:
data prep → tokenization → model training → evaluation →
inference. It is designed for biomedical abstracts,
clinical trials, and systematic reviews, producing concise
summaries that retain critical findings.

------------------------------------------------------------
2. FEATURES
------------------------------------------------------------
- Custom GPT-style decoder implemented with PyTorch
- Domain-specific corpora: PubMed & MS² medical review data
- End-to-end pipeline: preprocessing → training → evaluation
- Metrics: ROUGE-1/2/L and BLEU
- Reproducible checkpoints via Git LFS
- Interactive Jupyter notebook for experimentation
- Healthcare-aware preprocessing (cleaning references, etc.)

------------------------------------------------------------
3. REPOSITORY STRUCTURE
------------------------------------------------------------
MedicalTextSummarizationModel/
│
├── Dataset.zip
├── text_summ_model_training.ipynb
├── gpt_decoder_epoch1.pt
├── gpt_decoder_epoch2.pt
├── gpt_decoder_epoch3.pt
├── Medical_Txt_Summarization_Model_Documentation.pdf
├── requirements.txt
└── readme.txt

* Large files are tracked with Git LFS.

------------------------------------------------------------
4. QUICKSTART
------------------------------------------------------------
# Clone
git clone https://github.com/Fadyeskand224/MedicalTextSummarizationModel.git
cd MedicalTextSummarizationModel

# Enable Git LFS
git lfs install
git lfs pull

# Create environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate   (macOS/Linux)
.venv\Scripts\activate      (Windows)
pip install -r requirements.txt

# Unpack dataset
unzip Dataset.zip -d Dataset/

# Run notebook
jupyter notebook text_summ_model_training.ipynb

------------------------------------------------------------
5. SETUP & INSTALLATION
------------------------------------------------------------
Requirements:
- Python 3.9+
- pip, virtualenv or conda
- Jupyter
- (Optional) CUDA-enabled GPU

------------------------------------------------------------
6. TRAINING
------------------------------------------------------------
Steps:
1. Configure paths, hyperparameters, seeds
2. Load and clean datasets
3. Tokenize inputs
4. Initialize GPT-style decoder
5. Train for N epochs, save checkpoints
6. Evaluate with ROUGE/BLEU


------------------------------------------------------------
7. INFERENCE (GENERATE SUMMARIES)
------------------------------------------------------------
Example usage in Python:

import torch
from your_model_impl import YourSummarizerModel, YourTokenizer

ckpt_path = "gpt_decoder_epoch3.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = YourTokenizer(...)
model = YourSummarizerModel(...)
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state)
model.to(device).eval()

article = "OBJECTIVE: Evaluate the efficacy..."
inputs = tokenizer(article, return_tensors="pt", truncation=True).to(device)

with torch.no_grad():
    summary_ids = model.generate(**inputs, max_new_tokens=220)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(summary)

------------------------------------------------------------
8. DATASETS
------------------------------------------------------------
- PubMed Summarization dataset
- MS² Medical Systematic Reviews

Use for research/educational purposes only.

------------------------------------------------------------
9. MODEL CHECKPOINTS
------------------------------------------------------------
- gpt_decoder_epoch1.pt  (baseline)
- gpt_decoder_epoch2.pt  (improved)
- gpt_decoder_epoch3.pt  (best)

------------------------------------------------------------
10. CONFIGURATION & REPRODUCIBILITY
------------------------------------------------------------
Example hyperparameters:
- d_model: 768
- n_layers: 12
- n_heads: 12
- dropout: 0.1
- batch_size: 8
- max_input_tokens: 2048
- lr: 3e-5
- epochs: 3

------------------------------------------------------------
11. TROUBLESHOOTING
------------------------------------------------------------
- Large files missing → run "git lfs install && git lfs pull"
- CUDA OOM → lower batch size or sequence length
- Slow CPU training → use smaller configs or GPU

------------------------------------------------------------
12. FAQ
------------------------------------------------------------
Q: Can I fine-tune on my own data?
A: Yes, point the notebook to your dataset.

Q: Can I deploy as an API?
A: Yes, wrap inference in FastAPI/Flask.

------------------------------------------------------------
13. ROADMAP
------------------------------------------------------------
- Add Pegasus/BERTSUM baselines
- Release FastAPI demo service
- Package as pip module

------------------------------------------------------------
14. CONTRIBUTING
------------------------------------------------------------
Fork the repo, create a branch, submit a pull request.

------------------------------------------------------------
15. ACKNOWLEDGEMENTS
------------------------------------------------------------
- PubMed dataset
- MS² dataset
- PyTorch, Hugging Face

------------------------------------------------------------
16. CITATION
------------------------------------------------------------
@software{medical_text_summarization_model_2025,
  author  = {Eskandr, Fady},
  title   = {Medical Text Summarization Model},
  year    = {2025},
  url     = {https://github.com/Fadyeskand224/MedicalTextSummarizationModel}
}

------------------------------------------------------------
17. CONTACT
------------------------------------------------------------
Author: Fady Eskandr
GitHub: https://github.com/Fadyeskand224
LinkedIn: https://www.linkedin.com/in/fady-eskandr
