
---

### *Assignment 2 – Deep Learning for NLP*

<div align="center" style="text-align: center">
<h2>Neural Language Modeling – Character-Level LSTM</h2>
<p><strong>Training & Evaluating a Language Model from Scratch</strong></p>

<img src="https://img.shields.io/badge/Framework-PyTorch-red">
<img src="https://img.shields.io/badge/Dataset-Pride%20and%20Prejudice-green">
<img src="https://img.shields.io/badge/Model-LSTM-blue">
<img src="https://img.shields.io/badge/Metric-Perplexity-orange">
<img src="https://img.shields.io/badge/Notebook-Google%20Colab-yellow">
</div>

---

## **Contents**

* [Objective](#objective)
* [1. Environment Setup](#1-environment-setup)
* [2. Dataset Preparation](#2-dataset-preparation)
* [3. Vocabulary & Encoding](#3-vocabulary--encoding)
* [4. Model Architecture](#4-model-architecture)
* [5. Experiments](#5-experiments-underfit--best-fit--overfit)
* [6. Training & Validation Loss Plots](#6-training--validation-loss-plots)
* [7. Perplexity Scores](#7-perplexity-scores)
* [8. Saved Output Files](#8-saved-output-files)
* [Tools Used](#tools-used)
* [Observations](#observations)
* [Conclusion](#conclusion)

---

## **Objective**

The goal of this assignment was to build a **neural language model from scratch** using PyTorch and analyze how different model sizes behave in terms of:

* Learning ability
* Generalization
* Underfitting
* Overfitting
* Perplexity as an evaluation metric

The entire task follows a simple pipeline:

📕 **Novel (text)** → 🔢 **Characters → IDs** → 🧠 **LSTM** → ✨ **Next-character prediction**

---

# **1. Environment Setup**

I used **Google Colab** with GPU enabled.

```python
import torch
print(torch.__version__)
print(torch.cuda.get_device_name(0))
```

Additional packages:

```bash
pip install torch numpy matplotlib
```

This ensured a clean and reproducible environment.

---

# **2. Dataset Preparation**

The assignment required using ONLY the provided dataset:
📘 *Pride and Prejudice* by Jane Austen (Project Gutenberg).

Steps followed:

1. Uploaded the raw `.txt` file in Colab
2. Converted everything to lowercase
3. Removed headers/footers from Project Gutenberg
4. Extracted only the main story (using the phrase “it is a truth universally acknowledged” as the start)
5. Saved cleaned text as:

```
data/Pride_and_Prejudice_CLEANED.txt
```

This file was later used for vocabulary building and training.

---

# **3. Vocabulary & Encoding**

Since this is a **character-level** language model:

* Extracted all unique characters
* Built `char_to_idx` and `idx_to_char` mappings
* Converted entire book into a list of integer IDs
* Created sliding windows of length **100 characters** to form training sequences

Example:

```
X[i]   → 100 previous characters  
Y[i]   → next character to predict
```

This resulted in ~615k sequences.

A 90/10 split was used for **train/validation** sets.

---

# **4. Model Architecture**

I implemented a simple, student-friendly LSTM model:

```python
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed, hidden):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed)
        self.lstm = nn.LSTM(embed, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
```

No pretrained models, no high-level libraries — everything coded manually.

---

# **5. Experiments: Underfit • Best-Fit • Overfit**

To demonstrate understanding of model capacity, I trained three configurations.

---

## 🔵 **A. Underfitting**

**Model:** Extremely small

* Embedding = 16
* Hidden size = 32
* Epochs = 3

**Behavior:**

* Both losses high
* Very slow learning
* Perplexity remained high

✔ Classic underfitting.

---

## 🟢 **B. Best-Fit Model**

**Model:** Balanced size

* Embedding = 64
* Hidden = 128
* Epochs = 5

**Behavior:**

* Training & validation losses decreased smoothly
* Validation curve followed training curve closely
* Best perplexity
* No signs of under/over-fitting

✔ Selected as the final model.

---

## 🔴 **C. Overfitting**

**Model:** Larger network + tiny dataset

* Embedding = 128
* Hidden = 256
* Used only 1000 training sequences
* LR increased → 0.01
* Epochs = 7

**Behavior:**

* Train loss dropped fast
* Validation loss started increasing
* Perplexity worsened

✔ Clear overfitting (expected).

---

# **6. Training & Validation Loss Plots**

All three plots are saved inside:

```
plots/
│── underfit_loss.png
│── bestfit_loss.png
└── overfit_loss.png
```

Each graph clearly shows the behavior of the respective models.

---

# **7. Perplexity Scores**

Final perplexities (approximate):

| Model    | Perplexity       |
| -------- | ---------------- |
| Underfit | Very High (~50+) |
| Best-Fit | **~3.7**         |
| Overfit  | ~15–18           |

Perplexity confirms the best-fit model generalizes the best.

---

# **8. Saved Output Files**

All trained models:

```
models/
│── underfit_model.pt
│── bestfit_model.pt
└── overfit_model.pt
```

All plots:

```
plots/
│── underfit_loss.png
│── bestfit_loss.png
└── overfit_loss.png
```

Dataset:

```
data/Pride_and_Prejudice_CLEANED.txt
```

Notebook:

```
Assignment2_LanguageModel.ipynb
```

These meet the assignment deliverable requirements.

---

# **Tools Used**

| Tool             | Purpose              |
| ---------------- | -------------------- |
| **Python 3.12**  | Notebook runtime     |
| **PyTorch**      | Model implementation |
| **NumPy**        | Encoding & arrays    |
| **Matplotlib**   | Plotting loss curves |
| **Google Colab** | GPU training         |

---

# **Observations**

* Underfitting clearly occurs when the model is too small.
* Overfitting appears when the model is too large for a tiny dataset.
* The best-fit model performed consistently with low perplexity.
* Character-level LSTMs learn slowly but capture text style well.
* The training curves match expected theoretical behavior.

---

# **Conclusion**

I successfully implemented and evaluated a character-level neural language model from scratch using PyTorch.
I demonstrated:

* Underfitting
* Overfitting
* Best-fit training
* Perplexity computation
* Plot analysis
* Proper dataset preprocessing
* Model saving & reproducibility

This assignment improved my understanding of language modeling, sequence learning, and LSTM behavior on real text data.
