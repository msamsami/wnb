# WNB: General and weighted naive Bayes classifiers

![](https://img.shields.io/badge/version-v0.1.4-green)
![](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)

<p>
<img src="logo.png" alt="wnb logo" width="275" />
<br>
</p>

## Introduction
Naive Bayes is often recognized as one of the most popular classification algorithms in the machine learning community.
This package takes naive Bayes to a higher level by providing its implementations in more general and weighted settings.

### General naive Bayes
The issue with the well-known implementations of the naive Bayes algorithm (such as the ones in `sklearn.naive_bayes`
module) is that they assume a single distribution for the likelihoods of all features. Such an implementation can limit 
those who need to develop naive Bayes models with different distributions for feature likelihood. And enters **WNB** library!
It allows you to customize your naive Bayes model by specifying the likelihood distribution of each feature separately.
You can choose from a range of continuous and discrete probability distributions to design your classifier.

### Weighted naive Bayes
Although naive Bayes has many advantages such as simplicity and interpretability, its conditional independence assumption
rarely holds true in real-world applications. In order to alleviate its conditional independence assumption, many attribute
weighting naive Bayes (WNB) approaches have been proposed. Most of the proposed methods involve computationally demanding
optimization problems that do not allow for controlling the model's bias due to class imbalance. Minimum Log-likelihood
Difference WNB (MLD-WNB) is a novel weighting approach that optimizes the weights according to the Bayes optimal decision
rule and includes hyperparameters for controlling the model's bias. **WNB** library provides an efficient implementation
of gaussian MLD-WNB.

## Install
The easiest way to install the wnb library is by using `pip`:
```commandline
pip install git+https://github.com/msamsami/weighted-naive-bayes
```
This library is shipped as an all-in-one module implementation with minimalistic dependencies and requirements. 
Furthermore, it is fully compatible with Scikit-learn API.

## Getting started
Here, we show how you can use the library to train general and weighted naive Bayes classifiers. 

### General naive Bayes

A general naive Bayes model can be set up and used in four simple steps:

1. Import the `GeneralNB` class as well as `Distribution` enum class
```python
from wnb import GeneralNB, Distribution as D
```

2. Initialize a classifier and specify the likelihood distributions
```python
gnb = GeneralNB(distributions=[D.NORMAL, D.CATEGORICAL, D.EXPONENTIAL])
```

3. Fit the classifier to a training set (with three features)
```python
gnb.fit(X, y)
```

4. Predict on test data
```python
gnb.predict(X_test)
```

### Weighted naive Bayes

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
