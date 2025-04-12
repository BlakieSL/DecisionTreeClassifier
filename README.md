# Decision Tree Classifier Visualization

## Description
This project demonstrates how different parameters in a Decision Tree Classifier affect model performance and decision boundaries. Using scikit-learn's `make_moons` dataset, it trains classifiers with varying splitting criteria (Gini impurity and entropy) and maximum depths, then visualizes their decision boundaries alongside training/test accuracies.

## Features
- Generates a synthetic moons dataset with 10,000 samples and configurable noise.
- Splits data into training (80%) and test (20%) sets.
- Trains 8 Decision Tree models with combinations of:
  - Splitting criteria: `gini` or `entropy`
  - Maximum depths: `None` (unlimited), `3`, `5`, `10`
- Visualizes decision boundaries and data points using Matplotlib.
- Compares training and test accuracies to highlight overfitting/underfitting trends.

## Installation
Ensure the following dependencies are installed:
```bash
pip install numpy matplotlib scikit-learn
