# Recyclable & Household Waste Classifier

A deep learning project that classifies waste images into 30 categories using a custom CNN model built with TensorFlow/Keras.

## 🚀 Live Demo

**[Try it on Hugging Face Spaces →](https://huggingface.co/spaces/EiMonSoe/waste-classifier)**

Upload any waste item photo and the model returns the top 5 predicted categories with confidence scores.

---

## 📌 Project Overview

| Item | Detail |
|------|--------|
| Task | Multi-class image classification |
| Classes | 30 waste categories |
| Model | Custom CNN (4 convolutional blocks) |
| Input size | 128 × 128 RGB |
| Validation accuracy | ~69% |
| Framework | TensorFlow / Keras |
| Deployment | Gradio on Hugging Face Spaces |

---

## 🗂 Waste Categories

Aerosol cans, aluminum food cans, aluminum soda cans, cardboard boxes, cardboard packaging, clothing, coffee grounds, disposable plastic cutlery, eggshells, food waste, glass beverage bottles, glass cosmetic containers, glass food jars, magazines, newspaper, office paper, paper cups, plastic cup lids, plastic detergent bottles, plastic food containers, plastic shopping bags, plastic soda bottles, plastic straws, plastic trash bags, plastic water bottles, shoes, steel food cans, styrofoam cups, styrofoam food containers, tea bags.

---

## 🧠 Model Architecture

Custom CNN with 4 convolutional blocks:

- **Block 1** — Conv2D(32) + BatchNorm + MaxPooling
- **Block 2** — Conv2D(64) + BatchNorm + MaxPooling
- **Block 3** — Conv2D(128) + BatchNorm + MaxPooling
- **Block 4** — Conv2D(256) + BatchNorm + GlobalAveragePooling
- **Head** — Dense(256) + Dropout(0.4) + Dense(30, softmax)

---

## 📁 Repository Files

| File | Description |
|------|-------------|
| `Waste_Classification.ipynb` | Full training notebook — EDA, preprocessing, model training, evaluation |
| `app.py` | Gradio app deployed on Hugging Face Spaces |
| `label_map.json` | Mapping from class index (0–29) to category name |
| `requirements.txt` | Python dependencies |

> The trained model file (`waste_cnn_best.keras`) is hosted on Hugging Face Spaces due to file size.

---

## 📊 Dataset

**Recyclable and Household Waste Classification** from Kaggle  
[https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification)

- Images organised into `default` (studio) and `real_world` subfolders per category
- Split: 70% train / 15% validation / 15% test
- Data augmentation: random flip, brightness, contrast, saturation, hue

---

## ⚙️ How to Run Locally

```bash
git clone https://github.com/monsoee29-lang/waste-classifier
cd waste-classifier
pip install -r requirements.txt
python app.py
```

> You will also need to download `waste_cnn_best.keras` from the Hugging Face Space and place it in the same folder.

---

## 👩‍💻 Author

**Ei Mon Soe**
