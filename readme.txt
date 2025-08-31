# 🧠 Medical Text Summarization Model  

> A deep learning project for **abstractive summarization of medical literature and systematic reviews**, built with Transformers and trained on PubMed & MS² datasets.  

---

## 📖 Overview  

This repository contains a complete workflow for training, evaluating, and deploying a **Transformer-based summarization model** for the biomedical domain. The goal is to generate **concise, accurate summaries of complex medical documents** such as clinical studies, PubMed abstracts, and systematic reviews—helping researchers, clinicians, and students digest critical information faster.  

---

## ✨ Key Features  

- ⚙️ **Custom GPT-style decoder** built from scratch with PyTorch  
- 📚 **Domain-specific datasets**: PubMed & MS² reviews  
- 🔍 **End-to-end pipeline**: preprocessing → tokenization → training → evaluation → inference  
- 📊 **Evaluation metrics**: BLEU, ROUGE, F1  
- 💾 **Pre-trained checkpoints** (`gpt_decoder_epoch*.pt`) for reproducibility  
- 📓 **Interactive Jupyter notebook** (`text_summ_model_training.ipynb`) for experimentation  
- 📑 **Comprehensive documentation** included in PDF  

---

## 🗂️ Repository Structure  

MedicalTextSummarizationModel/
│
├── Dataset.zip # Original dataset bundle
├── text_summ_model_training.ipynb # Main training & evaluation notebook
├── gpt_decoder_epoch1.pt # Saved model checkpoint
├── gpt_decoder_epoch2.pt
├── gpt_decoder_epoch3.pt
├── Medical_Txt_Summarization_Model_Documentation.pdf
├── requirements.txt # Dependencies
└── readme.txt # (legacy notes)

yaml
Copy code

---

## 🚀 Getting Started  

### 1. Clone the repo  
```bash
git clone https://github.com/Fadyeskand224/MedicalTextSummarizationModel.git
cd MedicalTextSummarizationModel
2. Install dependencies
bash
Copy code
python3 -m venv .venv
source .venv/bin/activate   # on macOS/Linux
# .venv\Scripts\activate    # on Windows
pip install -r requirements.txt
3. Download datasets
Unzip the dataset bundle:

bash
Copy code
unzip Dataset.zip -d Dataset/
4. Train the model
Open the notebook:

bash
Copy code
jupyter notebook text_summ_model_training.ipynb
Follow the cells to preprocess data, train, and evaluate.

📊 Results
Training: 3 epochs on PubMed/MS² data

Metrics: ROUGE-1, ROUGE-2, ROUGE-L, BLEU

Findings: Model achieves strong abstractive summaries compared to baseline methods

📦 Model Checkpoints
gpt_decoder_epoch1.pt → baseline training run

gpt_decoder_epoch2.pt → improved performance with longer training

gpt_decoder_epoch3.pt → final, best performing checkpoint

🔮 Future Work
Fine-tune with Pegasus and BERTSUM baselines for comparison

Extend evaluation with clinical notes datasets (e.g., MIMIC-III)

Deploy as a REST API for real-time summarization

🤝 Contributing
Contributions are welcome! Fork the repo, create a branch, and open a pull request with improvements.

🙌 Acknowledgements
PubMed Summarization Dataset

MS² Dataset

Hugging Face Transformers

PyTorch
