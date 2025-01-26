# README: Credit Card Fraud Detection with Sampling and Model Evaluation

## Overview

This project focuses on detecting credit card fraud using machine learning techniques on an imbalanced dataset. The primary challenges addressed are the class imbalance and the evaluation of multiple models using various sampling techniques.

The script implements:
1. **Data Preprocessing**: Handling class imbalance using SMOTE.
2. **Sampling**: Creating different subsets of the resampled dataset.
3. **Model Training and Evaluation**: Assessing five machine learning models.
4. **Results Visualization**: Displaying an accuracy matrix and identifying the best sampling technique for each model.

---

## Features

- **Class Imbalance Handling**: Uses SMOTE to balance the dataset.
- **Multiple Sampling Techniques**: Five unique sampling methods are applied to the resampled dataset.
- **Diverse Machine Learning Models**:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - Support Vector Classifier (SVC)
  - K-Nearest Neighbors (KNN)
- **Accuracy Matrix**: Summarizes the performance of all models across sampling techniques.
- **Best Sampling Selection**: Highlights the optimal sampling method for each model.

---

## Installation

### Prerequisites
- Python 3.7 or later
- Required libraries:
  ```bash
  pip install pandas scikit-learn imbalanced-learn numpy
  ```

### Dataset
The project uses a CSV dataset containing credit card transactions with a `Class` column indicating fraud (1) or non-fraud (0). Place your dataset in the script's directory.

---

## How to Run

1. **Download the Script**  
   Clone the repository or save the script file locally.

2. **Prepare Your Dataset**  
   Ensure your dataset file is named (e.g., `Creditcard_data.csv`) and placed in the same directory as the script.

3. **Run the Script**  
   Execute the script with:
   ```bash
   python script_name.py
   ```

4. **Outputs**  
   - An accuracy matrix showing model performance across sampling techniques.
   - The best sampling technique for each model.
   - Results saved in `results_matrix_assignment_exact.csv`.

---

## Sampling Techniques

1. **Sampling1**: Random sampling with seed `45`.
2. **Sampling2**: Random sampling with seed `20`.
3. **Sampling3**: Systematic sampling (every nth row).
4. **Sampling4**: Random sampling with seed `55`.
5. **Sampling5**: Random sampling with seed `100`.

---

## Models

1. **Logistic Regression (LR)**
2. **Decision Tree (DT)**
3. **Random Forest (RF)**
4. **Support Vector Classifier (SVC)**
5. **K-Nearest Neighbors (KNN)**

---

## Outputs

### Accuracy Matrix
| Sampling    | Logistic Regression (LR) | Decision Tree (DT) | Random Forest (RF) | SVC    | KNN    |
|-------------|---------------------------|---------------------|---------------------|--------|--------|
| Sampling1   | 0.903333                  | 0.963333            | 0.990000            | 0.6600 | 0.8300 |
| Sampling2   | 0.916667                  | 0.973333            | 0.996667            | 0.6500 | 0.8567 |
| Sampling3   | 0.906114                  | 0.978166            | 0.993450            | 0.6769 | 0.8472 |
| Sampling4   | 0.926667                  | 0.986667            | 1.000000            | 0.6733 | 0.8067 |
| Sampling5   | 0.930000                  | 0.973333            | 0.996667            | 0.6500 | 0.8033 |

### Best Sampling Technique
| Model                  | Best Sampling Technique |
|------------------------|--------------------------|
| Logistic Regression    | Sampling5               |
| Decision Tree          | Sampling4               |
| Random Forest          | Sampling4               |
| Support Vector Machine | Sampling3               |
| K-Nearest Neighbors    | Sampling2               |

---

## Notes

- **Convergence Warning**: For Logistic Regression, the `lbfgs` solver may fail to converge. Consider scaling data or increasing `max_iter`.
- **Customizing Sample Size**: Adjust the `sample_size` variable in the script to modify the size of sampled datasets.
- **Data Preprocessing**: Ensure numerical data is properly formatted before running the script.

---

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this project as per the license terms.
