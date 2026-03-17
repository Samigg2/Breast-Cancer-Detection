# Breast Cancer Detection Using a Decision Tree

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
````

### 2. Manual Gini Calculation

Root node had 357 benign and 212 malignant samples, total 569.

**Parent Gini:**

```text
1 - (357/569)^2 - (212/569)^2 ≈ 0.468
```

Tested a split on `radius_mean ≤ 14.5`:

* **Left node**: 300 benign, 50 malignant → Gini ≈ 0.245
* **Right node**: 57 benign, 162 malignant → Gini ≈ 0.384

**Weighted average after split:**

```text
(350/569 * 0.245) + (219/569 * 0.384) ≈ 0.299
```

The impurity dropped from 0.468 to 0.299, which means the split created cleaner groups.

---

### 3. Depth Experiment

Trained trees with different depth limits:

| Max Depth | Train Accuracy | Test Accuracy |
| --------- | -------------- | ------------- |
| 2         | 92%            | 91%           |
| 5         | 98%            | 94%           |
| 10        | 100%           | 93%           |
| None      | 100%           | 91%           |

Deeper trees memorize training data perfectly, but test accuracy eventually drops. That's **overfitting**.

---

### 4. Feature Importance

Model ranked features by how much they influence predictions:

* concave points mean: 0.32
* radius mean: 0.21
* perimeter mean: 0.15

Makes sense—irregular shapes are often linked to malignancy. But correlation isn't causation.

---

### 5. Where It Messes Up

Overall accuracy was around 95%. But a few malignant tumors were predicted as benign. In a real hospital, **false negatives are worse than false positives**.

If I kept working on this, I'd try to tilt the model to be safer—favor fewer missed cancers even if that means more false alarms.

---

## What I Actually Learned

Before this, decision trees felt like magic boxes. After calculating impurities manually and watching how depth changes performance, they feel less mysterious. Just math. Repeated simple math on real numbers from real tumors.

The dataset here is clean. Real medical data would be messier—missing values, inconsistent measurements, noise. That's probably where the interesting problems actually are.

I don't just want to run models. I want to understand the math underneath them well enough that when I look at a problem like women in Ethiopia getting diagnosed too late, I'll know what the data actually says.

---

## How to Run This

### 1. Clone the repo

```bash
git clone https://github.com/Samigg2/Breast-Cancer-Detection.git
cd Breast-Cancer-Detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open `Decision Trees with Scikitlearn.ipynb` in **Jupyter**, **VS Code**, or **Google Colab** and run cells sequentially.

---

## requirements.txt

Create a file called `requirements.txt` with exactly:

```text
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

---

## Possible Improvements

* Tune hyperparameters like `max_depth` and `min_samples_split`
* Use cross-validation for more reliable evaluation
* Compare with Random Forest, SVM, or Logistic Regression
* Try cost-sensitive learning to penalize false negatives more heavily

---

## Final Thought

This project didn't change the world. But it changed how I think about machine learning. These models aren't magic—they're just math, applied carefully, on real data from real people.

That's worth understanding deeply.

```
