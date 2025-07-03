<div align="center">
<img src="https://raw.githubusercontent.com/msamsami/wnb/main/docs/logo.png" alt="wnb logo" width="275" />
</div>

<div align="center"> <b>General and weighted naive Bayes classifiers</b> </div>
<div align="center">Scikit-learn-compatible</div> <br>

<div align="center">

![Lastest Release](https://img.shields.io/badge/release-v0.8.1-green)
[![PyPI Version](https://img.shields.io/pypi/v/wnb)](https://pypi.org/project/wnb/)
![Python Versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)<br>
![GitHub Workflow Status (build)](https://github.com/msamsami/wnb/actions/workflows/build.yml/badge.svg)
[![Coverage](https://codecov.io/gh/msamsami/wnb/graph/badge.svg?token=74EIO9XQUY)](https://codecov.io/gh/msamsami/wnb)
![PyPI License](https://img.shields.io/pypi/l/wnb)
[![PyPi Downloads](https://static.pepy.tech/badge/wnb)](https://pepy.tech/project/wnb)

</div>

## Introduction
Naive Bayes is a widely used classification algorithm known for its simplicity and efficiency. This package takes naive Bayes to a higher level by providing more flexible and weighted variants, making it suitable for a broader range of applications.

### General naive Bayes
Most standard implementations, such as those in `sklearn.naive_bayes`, assume a single distribution type for all feature likelihoods. This can be restrictive when dealing with mixed data types. **WNB** overcomes this limitation by allowing users to specify different probability distributions for each feature individually. You can select from a variety of continuous and discrete distributions, enabling greater customization and improved model performance.

### Weighted naive Bayes
While naive Bayes is simple and interpretable, its conditional independence assumption often fails in real-world scenarios. To address this, various attribute-weighted naive Bayes methods exist, but most are computationally expensive and lack mechanisms for handling class imbalance.

**WNB** package provides an optimized implementation of *Minimum Log-likelihood Difference Wighted Naive Bayes* (MLD-WNB), a novel approach that optimizes feature weights using the Bayes optimal decision rule. It also introduces hyperparameters for controlling model bias, making it more robust for imbalanced classification.

## Installation
This library is shipped as an all-in-one module implementation with minimalistic dependencies and requirements. Furthermore, it fully **adheres to Scikit-learn API** ❤️.

### Prerequisites
Ensure that Python 3.8 or higher is installed on your machine before installing **WNB**.

### PyPi
```bash
pip install wnb
```

### uv
```bash
uv add wnb
```

## Getting started ⚡️
Here, we show how you can use the library to train general (mixed) and weighted naive Bayes classifiers.

### General naive Bayes

A general naive Bayes model can be set up and used in four simple steps:

1. Import the `GeneralNB` class as well as `Distribution` enum class
```python
from wnb import GeneralNB, Distribution as D
```

2. Initialize a classifier with likelihood distributions specified
```python
clf = GeneralNB([D.NORMAL, D.CATEGORICAL, D.EXPONENTIAL, D.EXPONENTIAL])
```
or
```python
# Columns not explicitly specified will default to Gaussian (normal) distribution
clf = GeneralNB(
    distributions=[
        (D.CATEGORICAL, [1]),
        (D.EXPONENTIAL, ["col3", "col4"]),
    ],
)
```

3. Fit the classifier to a training set (with four features)
```python
clf.fit(X_train, y_train)
```

4. Predict on test data
```python
clf.predict(X_test)
```

### Weighted naive Bayes

An MLD-WNB model can be set up and used in four simple steps:

1. Import the `GaussianWNB` class
```python
from wnb import GaussianWNB
```

2. Initialize a classifier
```python
clf = GaussianWNB(max_iter=25, step_size=1e-2, penalty="l2")
```

3. Fit the classifier to a training set
```python
clf.fit(X_train, y_train)
```

4. Predict on test data
```python
clf.predict(X_test)
```

## Compatibility with Scikit-learn 🤝

The **wnb** library fully adheres to the Scikit-learn API, ensuring seamless integration with other Scikit-learn components and workflows. This means that users familiar with Scikit-learn will find the WNB classifiers intuitive to use.

Both Scikit-learn classifiers and WNB classifiers share these well-known methods:

- `fit(X, y)`
- `predict(X)`
- `predict_proba(X)`
- `predict_log_proba(X)`
- `predict_joint_log_proba(X)`
- `score(X, y)`
- `get_params()`
- `set_params(**params)`
- etc.

By maintaining this consistency, WNB classifiers can be easily incorporated into existing machine learning pipelines and processes.

## Benchmarks 📊
We conducted benchmarks on four datasets, [Wine](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset), [Iris](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset), [Digits](https://scikit-learn.org/stable/datasets/toy_dataset.html#optical-recognition-of-handwritten-digits-dataset), and [Breast Cancer](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset), to evaluate the performance of WNB classifiers and compare them with their Scikit-learn counterpart, `GaussianNB`. The results show that WNB classifiers generally perform better in certain cases.

| Dataset          | Scikit-learn Classifier | Accuracy | WNB Classifier | Accuracy  |
|------------------|-------------------------|----------|----------------|-----------|
| Wine             | GaussianNB              | 0.9749    | GeneralNB      | **0.9812**     |
| Iris             | GaussianNB              | 0.9556    | GeneralNB      | **0.9602**     |
| Digits           | GaussianNB              | 0.8372    | GeneralNB      | **0.8905**     |
| Breast Cancer    | GaussianNB              | 0.9389    | GaussianWNB    | **0.9519**     |

These benchmarks highlight the potential of WNB classifiers to provide better performance in certain scenarios by allowing more flexibility in the choice of distributions and incorporating weighting strategies.

The scripts used to generate these benchmark results are available in the _benchmarks/_ directory.

## Support us 💡
You can support the project in the following ways:

⭐ Star WNB on GitHub (click the star button in the top right corner)

💡 Provide your feedback or propose ideas in the [Issues section](https://github.com/msamsami/wnb/issues)

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
  howpublished = {\url{https://github.com/msamsami/wnb}},
}
```
