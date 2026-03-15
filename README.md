# Breast Cancer Detection Using a Decision Tree

**Author**: Sami  
**Course**: Fundamentals of AI and Machine Learning  
**Tools**: Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn  

---

## Why This Project Exists

This wasn't just a class assignment. I kept seeing news reports from Ethiopia—women finding out they had breast cancer, but only after it had already spread. Mothers. One interview stuck with me: a doctor said if they'd caught it earlier, treatment would have been simpler. That hit me.

I started wondering: could data catch what doctors miss? Patterns before symptoms show up?

Obviously a student project won't solve this. But I wanted to understand how these models actually work under the hood—not just run code, but see what the math is doing. So I built a decision tree classifier on real tumor data.

This repository is that project.

---

## What's Inside

- **Decision Trees with Scikitlearn.ipynb** – Full implementation with manual Gini calculation and depth experiments  
- **README.md** – This file  
- **requirements.txt** – All libraries needed to run this  

---

## The Dataset

I used the **Breast Cancer Wisconsin (Diagnostic) dataset** (built into Scikit-learn). Each row is measurements from a breast mass image:

- radius  
- texture  
- perimeter  
- area  
- smoothness  
- concavity  
- symmetry  
- (and more)  

Two labels: **malignant** (cancerous) or **benign** (not).  

While exploring, I noticed some variables are basically the same thing measured differently—radius and perimeter both describe size. So the model probably leans on one and ignores the other.

---

## What I Actually Did

### 1. Basic Pipeline

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
