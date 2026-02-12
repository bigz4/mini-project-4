# StyleSort – Fashion Classification Project

## 1. Business Problem Description

**Company Context – StyleSort**

StyleSort is an online fashion retailer processing over **100,000 new product listings per month**. The company is currently experiencing a **32% return rate**, significantly higher than the industry average of 20%. Internal customer surveys reveal that approximately **40% of returns occur because the item was “not what the customer expected.”**

A major contributor to this issue is **product misclassification** within StyleSort’s online catalog. Examples of costly errors include:

* A formal **Shirt** listed as a casual **T-shirt**
* **Ankle boots** categorized as **Sandals**
* A **Coat** listed as a **Pullover**, leading to incorrect warmth and sizing expectations
* **Sneakers** confused with **Ankle boots**, affecting price and use-case expectations

These errors lead to customer dissatisfaction, increased logistics costs, negative reviews, and lost revenue.

**Business Goal**

StyleSort hired our team to build an automated image classification system that can:

1. Accurately classify product images into one of **10 clothing categories**
2. Provide **confidence scores** to flag low-confidence predictions for human review
3. Identify **frequently confused category pairs** to guide catalog and photography improvements

The system must achieve **>85% accuracy**, process thousands of listings per day efficiently, and provide actionable insights for business decision-making.

---

## 2. Approach and Methodology

Our approach follows a standard machine learning pipeline:

1. **Data Preparation**

   * Used a labeled fashion image dataset (Fashion-MNIST–style grayscale images).
   * Normalized pixel values to improve training stability.
   * Split the dataset into training and test sets.

2. **Model Development**

   * Implemented a neural network using PyTorch.
   * Experimented with different layers and activation functions.
   * Used cross-entropy loss for multi-class classification.

3. **Training & Evaluation**

   * Trained the model over multiple epochs.
   * Monitored training loss and accuracy per epoch.
   * Evaluated performance on a held-out test set.

4. **Iteration & Debugging**

   * Adjusted learning rate and architecture based on observed performance.
   * Ensured correct device usage (CPU/GPU).

---

## 3. Model Architecture Description

The final model consists of:

* **Input Layer**: Flattened 28×28 grayscale image (784 features)
* **Hidden Layers**:

  * Fully connected layer with ReLU activation
  * Additional fully connected layer to improve feature learning
* **Output Layer**:

  * Fully connected layer with 10 output neurons (one per clothing class)

**Key Design Choices**:

* ReLU activation was chosen to mitigate vanishing gradients.
* A simple feedforward architecture was used to balance performance and interpretability.

---

## 4. Results Summary and Key Metrics

Our best-performing model exceeded the business accuracy threshold required by StyleSort.

**Key Metrics (Best Model):**

* **Training Accuracy:** ~88–92%
* **Test Accuracy:** **>85%** (meeting business requirements)
* **Loss Function:** Cross-Entropy Loss
* **Optimizer:** Adam / AdamW (best-performing configuration)

**Observations:**

* The gap between training and test accuracy was small, indicating good generalization.
* Most errors occurred between visually similar categories such as **Shirt vs T-shirt** and **Sneaker vs Ankle boot**.

Additional analyses (confusion matrix, misclassification examples, and training curves) were generated to better understand model behavior and business impact.

---

## 5. Business Recommendations

Based on the model’s performance and error analysis, we recommend the following actions for StyleSort:

1. **Human-in-the-Loop Deployment**
   Use the model to automatically categorize products, but flag predictions below a confidence threshold (e.g., 80%) for human review. This balances automation with accuracy.

2. **Focus on High-Cost Errors**
   Prioritize reducing misclassifications between high-impact category pairs such as **Shirt ↔ T-shirt** and **Coat ↔ Pullover**, as these lead to the most costly returns.

3. **Improve Product Photography Guidelines**
   Confusion between similar categories suggests a need for clearer product images and standardized angles.

4. **Model Improvements**
   Transition from a fully connected network to a **Convolutional Neural Network (CNN)** to better capture spatial features and further reduce error rates.

5. **Continuous Monitoring**
   Track cost-weighted accuracy and confusion trends over time to ensure the model continues to deliver business value.

---

## 6. Setup and Running Instructions

### Requirements

* Python 3.9+
* PyTorch
* torchvision
* numpy
* matplotlib

### Installation

```bash
pip install -r requirements.txt
```

### Running the Project

1. Clone the repository:

```bash
git clone https://github.com/bigz4/mini-project-4
```

2. Run all cells in fashion_classifier.ipynb

---

## 7. Authors

* **Aristide Kanamugire**: Analysis and Report
* **Brendan Zapf**: Analysis and git structure


---

