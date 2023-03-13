# MLD-WNB

![](https://img.shields.io/badge/version-v0.0.5-green)
![](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)

<p>
<img src="logo.png" alt="MLD-WNB logo" />
<br>
</p>

## Introduction
Naive Bayes (NB) is recognized as one of the most popular classification algorithms in the machine learning community, but its conditional independence assumption rarely holds true in real-world applications. In order to alleviate its conditional independence assumption, many attribute weighting NB (WNB) approaches have been proposed. Most of the proposed methods involve computationally demanding optimization problems that do not allow for controlling the model's bias due to class imbalance.

**Minimum Log-likelihood Difference WNB (MLD-WNB)** is a novel weighting approach that optimizes the weights according to the Bayes optimal decision rule and includes hyperparameters for controlling the model's bias. `wnb` library provides an efficient implementation of MLD-WNB which is compatible with Scikit-learn API.

## Install
The easiest way to install the wnb library is by using `pip`:
```commandline
pip install git+https://github.com/msamsami/weighted-naive-bayes
```
This library is shipped as an all-in-one module implementation with minimalistic dependencies and requirements.

## Getting started
An MLD-WNB model can be set up and used in four simple steps:
1. Import the `GaussianWNB` class
```python
from wnb import GaussianWNB
```

2. Initialize a classifier
```python
wnb = GaussianWNB(step_size=1e-2, max_iter=25)
```

3. Fit the classifier to a training set
```python
wnb.fit(X, y)
```

4. Predict on test data
```python
wnb.predict(x_test)
```
