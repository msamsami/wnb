from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.naive_bayes import GaussianNB

from benchmarks.utils import benchmark
from wnb import Distribution as D
from wnb import GaussianWNB, GeneralNB

MAX_ITER = 100


def benchmark_wine() -> None:
    X, y = load_wine(return_X_y=True)

    clf_wnb = GeneralNB(distributions=[D.LOGNORMAL] * X.shape[1])
    clf_sklearn = GaussianNB()
    score_wnb, score_sklearn = benchmark(X, y, clf_wnb, clf_sklearn, MAX_ITER)

    print("wine dataset | sklearn | GaussianNB >> score >>", score_sklearn)
    print("wine dataset | wnb     | GeneralNB  >> score >>", score_wnb, "\n")


def benchmark_iris() -> None:
    X, y = load_iris(return_X_y=True)

    clf_wnb = GeneralNB(distributions=[D.EXPONENTIAL, D.RAYLEIGH, D.NORMAL, D.NORMAL])
    clf_sklearn = GaussianNB()
    score_wnb, score_sklearn = benchmark(X, y, clf_wnb, clf_sklearn, MAX_ITER)

    print("iris dataset | sklearn | GaussianNB >> score >>", score_sklearn)
    print("iris dataset | wnb     | GeneralNB  >> score >>", score_wnb, "\n")


def benchmark_digits() -> None:
    X, y = load_digits(return_X_y=True)

    clf_wnb = GeneralNB(distributions=[D.POISSON] * X.shape[1])
    clf_sklearn = GaussianNB()

    score_wnb, score_sklearn = benchmark(X, y, clf_wnb, clf_sklearn, MAX_ITER)

    print("digits dataset | sklearn | GaussianNB >> score >>", score_sklearn)
    print("digits dataset | wnb     | GeneralNB  >> score >>", score_wnb, "\n")


def benchmark_breast_cancer() -> None:
    X, y = load_breast_cancer(return_X_y=True)

    clf_wnb = GaussianWNB(max_iter=30, step_size=0.01, C=1.5, var_smoothing=1e-12)
    clf_sklearn = GaussianNB()
    score_wnb, score_sklearn = benchmark(X, y, clf_wnb, clf_sklearn, MAX_ITER)

    print("breast cancer dataset | sklearn | GaussianNB  >> score >>", score_sklearn)
    print("breast cancer dataset | wnb     | GaussianWNB >> score >>", score_wnb, "\n")


if __name__ == "__main__":
    benchmark_wine()
    benchmark_iris()
    benchmark_digits()
    benchmark_breast_cancer()
