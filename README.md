<div align="center">
<img src="https://raw.githubusercontent.com/msamsami/weighted-naive-bayes/main/docs/logo.png" alt="wnb logo" width="275" />
</div>

<div align="center"> <b>General and weighted naive Bayes classifiers</b> </div>
<div align="center">Scikit-learn-compatible</div> <br>

<div align="center">

![Lastest Release](https://img.shields.io/badge/release-v0.2.5-green)
[![PyPI Version](https://img.shields.io/pypi/v/wnb)](https://pypi.org/project/wnb/)
![Python Versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)<br>
![GitHub Workflow Status (build)](https://github.com/msamsami/weighted-naive-bayes/actions/workflows/python-publish.yml/badge.svg)
![PyPI License](https://img.shields.io/pypi/l/wnb)
[![PyPi Downloads](https://static.pepy.tech/badge/wnb)](https://pepy.tech/project/wnb)

</div>

## Introduction
Naive Bayes is often recognized as one of the most popular classification algorithms in the machine learning community. This package takes naive Bayes to a higher level by providing its implementations in more general and weighted settings.

### General naive Bayes
The issue with the well-known implementations of the naive Bayes algorithm (such as the ones in `sklearn.naive_bayes` module) is that they assume a single distribution for the likelihoods of all features. Such an implementation can limit those who need to develop naive Bayes models with different distributions for feature likelihood. And enters **WNB** library! It allows you to customize your naive Bayes model by specifying the likelihood distribution of each feature separately. You can choose from a range of continuous and discrete probability distributions to design your classifier.

### Weighted naive Bayes
Although naive Bayes has many advantages such as simplicity and interpretability, its conditional independence assumption rarely holds true in real-world applications. In order to alleviate its conditional independence assumption, many attribute weighting naive Bayes (WNB) approaches have been proposed. Most of the proposed methods involve computationally demanding optimization problems that do not allow for controlling the model's bias due to class imbalance. Minimum Log-likelihood Difference WNB (MLD-WNB) is a novel weighting approach that optimizes the weights according to the Bayes optimal decision rule and includes hyperparameters for controlling the model's bias. **WNB** library provides an efficient implementation of gaussian MLD-WNB.

## Installation
The easiest way to install the **wnb** library is by using `pip`:
```
pip install wnb
```
This library is shipped as an all-in-one module implementation with minimalistic dependencies and requirements. Furthermore, it fully **adheres to Scikit-learn API** ❤️.

## Getting started ⚡️
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
wnb = GaussianWNB(max_iter=25, step_size=1e-2, penalty="l2")
```

3. Fit the classifier to a training set
```python
wnb.fit(X, y)
```

4. Predict on test data
```python
wnb.predict(x_test)
```

## Compatibility with Scikit-learn

The **wnb** library fully adheres to the Scikit-learn API, ensuring seamless integration with other Scikit-learn components and workflows. This means that users familiar with Scikit-learn will find the WNB classifiers intuitive to use.

Both Scikit-learn classifiers and WNB classifiers share these well-known methods:

- `fit(X, y)`
- `predict(X)`
- `predict_proba(X)`
- `predict_log_proba(X)`
- `score(X, y)`
- `get_params()`
- `set_params(**params)`
- etc.

By maintaining this consistency, WNB classifiers can be easily incorporated into existing machine learning pipelines and processes.

## Benchmarks
We conducted benchmarks on three datasets, [Breast Cancer](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset), [Digits](https://scikit-learn.org/stable/datasets/toy_dataset.html#optical-recognition-of-handwritten-digits-dataset), and [Wine](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset), to evaluate the performance of WNB classifiers and compare them with their Scikit-learn counterpart, `GaussianNB`. The results show that WNB classifiers generally perform better in certain cases.

| Dataset          | Scikit-learn Classifier | Accuracy | WNB Classifier | Accuracy  |
|------------------|-------------------------|----------|----------------|-----------|
| Breast Cancer    | GaussianNB              | 0.939    | GaussianWNB    | **0.951**     |
| Digits           | GaussianNB              | 0.838    | GeneralNB      | **0.889**     |
| Wine             | GaussianNB              | 0.974    | GeneralNB      | **0.981**     |

These benchmarks highlight the potential of WNB classifiers to provide better performance in certain scenarios by allowing more flexibility in the choice of distributions and incorporating weighting strategies.

The benchmark scripts used to obtain these results can be found under _tests/benchmarks/_ directory.

## Tests
To run the tests, make sure to clone the repository and install the development requirements:
```
pip install -r requirements_dev.txt
```

Then, run pytest:
```
pytest
```

## Support us 🤝
You can support the project in the following ways:

⭐ Star WNB on GitHub (click the star button in the top right corner)

💡 Provide your feedback or propose ideas in the [Issues section](https://github.com/msamsami/weighted-naive-bayes/issues)

📰 Post about WNB on LinkedIn or other platforms


## Citation 📚
If you utilize this repository, please consider citing it with:

```
@misc{wnb,
  author = {Mohammd Mehdi Samsami},
  title = {WNB: General and weighted naive Bayes classifiers},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/msamsami/weighted-naive-bayes}},
}
```