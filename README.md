# 🧠 Transformer Attention Visualization App

![Streamlit](https://img.shields.io/badge/Made%20With-Streamlit-red?style=flat-square)
![Hugging Face](https://img.shields.io/badge/Powered%20By-Hugging%20Face-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

## 🚀 Overview
This **Transformer Attention Visualization App** allows users to explore **how transformer models (BERT, GPT-2, RoBERTa, etc.) distribute attention across words in a sentence**.

Users can:
- Compare **different models** side by side.
- View **interactive heatmaps, word clouds, attention graphs, and arc diagrams**.
- Highlight **important words based on attention scores**.
- Analyze **special token trends ([CLS], [SEP], etc.) across layers**.

---

## 🎯 Features
✅ **Compare Models:** Choose two transformer models and compare their attention.  
✅ **Attention Heatmaps:** See how attention is distributed across words.  
✅ **Word Cloud:** Visualize important tokens based on attention strength.  
✅ **Attention Arcs:** Explore long-range dependencies in sentences.  
✅ **Interactive Graphs:** Create force-directed attention graphs.  
✅ **Layer & Head Selection:** Select specific layers and attention heads for analysis.  

---

## 🎥 Demo
🚀 **Live App on Streamlit:** [Click Here](https://attentionviz.streamlit.app/)  


---

## 📦 Installation
To run the project locally, follow these steps:

### Clone the Repository

git clone https://github.com/syedmujtaba-10/Attention-Viz
cd Attention-Viz

### Install Dependencies
Make sure you have Python >=3.8 installed. Then run:

pip install -r requirements.txt

### Run the app

streamlit run app.py
