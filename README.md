# Iris Classification - ICMC/USP

## Overview

This project implements a machine learning pipeline to classify the Iris dataset using various models. The Iris dataset is a well-known dataset in the field of machine learning, consisting of 150 samples from three species of Iris flowers (`Iris setosa`, `Iris versicolor`, `Iris virginica`). Each sample has four features: `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, and `PetalWidthCm`. The goal of this project is to apply machine learning algorithms for classification tasks and compare their performance.

## Project Structure

This repository contains the following files:

- `main.py`: The main Python script that loads the Iris dataset, preprocesses the data, trains multiple machine learning models, and displays results.
- `iris.data`: The Iris dataset used in the project (in CSV format).
- `README.md`: This file, providing an overview and instructions for the project.

## Machine Learning Models Used

In this project, two machine learning models are implemented:

1. **Support Vector Machine (SVM)**:
   - A supervised learning model that is used for classification tasks, particularly useful for linear and non-linear data.

2. **Linear Regression**:
   - Although typically used for regression tasks, this model is used here for comparison to evaluate its performance on a classification task.

## Requirements

The following Python packages are required to run the code:

- `pandas`: For data manipulation and analysis.
- `scikit-learn`: For implementing machine learning algorithms.
- `colorama`: For colored terminal output.
- `tabulate`: For displaying classification reports in a tabular format.

You can install these dependencies by running:

```bash
pip install -r requirements.txt
