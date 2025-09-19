# FinQA-T5
![Transformers](https://img.shields.io/badge/🤗%20Transformers-orange?logo=huggingface&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)
![LoRA](https://img.shields.io/badge/LoRA-low--rank%20adaptation-blue)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?&logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-%23150458.svg?&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?&logo=plotly&logoColor=blue)
![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)
![Topic Modeling](https://img.shields.io/badge/Topic%20Modeling-NLP-green)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Finance 500K – Data Analysis, Topic Modeling & Transformer Training

This repository contains experiments and analysis performed on the **Finance 500K dataset** from Hugging Face. The workflow includes dataset preprocessing, exploratory data analysis, topic modeling, transformer training (with LoRA), and Retrieval-Augmented Generation (RAG) experiments.

## 🚀 Project Overview

1. **Dataset Preprocessing**  
   - Original dataset: [Finance-Instruct-500k by Josephgflowers](https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k)  
   - Additional resource: [Finance-500k-train collection](https://huggingface.co/collections/kartikayluthra01/finance-500k-train-68cbf232fceb092f08fb788e)  
   - To work efficiently on an **8 GB RAM** setup, the dataset was sampled down to **1,000 rows**.  
   - This smaller dataset was used for both analysis and model training.

2. **Exploratory Data Analysis (EDA)** – *`data_analysis.ipynb`*  
   - Analyzed **average user query lengths** and **average assistant response lengths** in terms of tokens.  
   - Visualized distributions, summary statistics, etc.  

3. **Topic Modeling** – *`topics.ipynb`*  
   - Performed topic modeling to identify the most frequent financial discussion themes in the dataset.  

4. **Transformer Training** – *`transformer.ipynb`*  
   - Fine-tuned a **Google T5 Mini (seq2seq model)** using **LoRA (Low-Rank Adaptation)** for efficient training.  
   - Training conducted on the reduced dataset to remain within memory constraints.  
   - Model checkpoints were published on Hugging Face (see links below).

5. **RAG Experiments** – *`rag.ipynb`*  
   - Conducted **Retrieval-Augmented Generation (RAG)** experiments.  
   - Explored how adding retrieval affects response relevance and contextuality.

---

## 📂 Repository Structure
```bash
notebooks/
├── data_analysis.ipynb # EDA: token lengths, query/response statistics
├── rag.ipynb # Retrieval-Augmented Generation experiments
├── topics.ipynb # Topic modeling
├── train_transformers.ipynb # Transformer training with T5 Mini + LoRA

scripts/
├── convert_to_csv.py # Convert dataset to CSV format
├── download_dataset.py # Script to download dataset
├── tohuggingface.py # Script to push dataset/model to Hugging Face
```

## 📊 Results & Insights

- **Topic Modeling**: Discovered dominant financial topics in the dataset.  
- **Token Lengths**: Benchmarked average user queries vs. assistant answers.  
- **LoRA Fine-tuning**: Enabled training T5 Mini efficiently with limited hardware.  
- **RAG**: Showed improvements in contextuality when augmented with retrieval.

---

## 🤗 Hugging Face Resources

- **Dataset**: [Finance-Instruct-500k](https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k)  
- **Collection**: [Finance-500k-train collection](https://huggingface.co/collections/kartikayluthra01/finance-500k-train-68cbf232fceb092f08fb788e)  
- **Model Checkpoints**: [Your Model Checkpoints Link](https://huggingface.co/collections/kartikayluthra01/finance-500k-train-68cbf232fceb092f08fb788e)  

---
## ⚙️ Technologies Used  

- [Hugging Face Datasets](https://huggingface.co/datasets)  
- [Transformers](https://huggingface.co/transformers)  
- [LoRA](https://arxiv.org/abs/2106.09685)  
- [scikit-learn](https://scikit-learn.org/) (for topic modeling)  
- [PyTorch](https://pytorch.org/)  
- [pandas](https://pandas.pydata.org/)  
- [NumPy](https://numpy.org/)  
- [FAISS](https://faiss.ai/) (for similarity search in RAG)  
