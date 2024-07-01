from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.naive_bayes import GaussianNB

from tests.benchmarks.utils import benchmark
from wnb import Distribution as D
from wnb import GaussianWNB, GeneralNB

MAX_ITER = 100


def benchmark_breast_cancer():
    breast_cancer = load_breast_cancer()
    X = breast_cancer["data"]
    y = breast_cancer["target"]

    clf_wnb = GaussianWNB(max_iter=20, step_size=0.01, C=1.5)
    clf_sklearn = GaussianNB()
    score_wnb, score_sklearn = benchmark(X, y, clf_wnb, clf_sklearn, MAX_ITER)

    print("breast cancer dataset | sklearn | GaussianNB  >> score >>", score_sklearn)
    print("breast cancer dataset | wnb     | GaussianWNB >> score >>", score_wnb, "\n")


def benchmark_digits():
    digits = load_digits()
    X = digits["data"]
    y = digits["target"]

    clf_wnb = GeneralNB(distributions=[D.POISSON] * X.shape[1])
    clf_sklearn = GaussianNB()
    score_wnb, score_sklearn = benchmark(X, y, clf_wnb, clf_sklearn, MAX_ITER)

    print("digits dataset | sklearn | GaussianNB >> score >>", score_sklearn)
    print("digits dataset | wnb     | GeneralNB  >> score >>", score_wnb, "\n")


def benchmark_wine():
    wine = load_wine()
    X = wine["data"]
    y = wine["target"]

    clf_wnb = GeneralNB(distributions=[D.LOGNORMAL] * X.shape[1])
    clf_sklearn = GaussianNB()
    score_wnb, score_sklearn = benchmark(X, y, clf_wnb, clf_sklearn, MAX_ITER)

    print("wine dataset | sklearn | GaussianNB >> score >>", score_sklearn)
    print("wine dataset | wnb     | GeneralNB  >> score >>", score_wnb, "\n")


if __name__ == "__main__":
    benchmark_breast_cancer()
    benchmark_digits()
    benchmark_wine()
