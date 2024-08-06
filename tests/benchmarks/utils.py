from __future__ import annotations

import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import train_test_split

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda iterable, *args, **kwargs: iterable

warnings.filterwarnings("ignore")

__all__ = ("benchmark",)


def compare_score(X, y, wnb, sklearn, random_state: int, test_size: float) -> tuple[float, float]:
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train and score wnb classifier
    clf_wnb = clone(wnb)
    clf_wnb.fit(X_train, y_train)

    # Train and score sklearn classifier
    clf_sklearn = clone(sklearn)
    clf_sklearn.fit(X_train, y_train)

    return clf_wnb.score(X_test, y_test), clf_sklearn.score(X_test, y_test)


def benchmark(X, y, wnb, sklearn, max_iter: int = 50, test_size: float = 0.33) -> tuple[float, float]:
    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(compare_score)(*param)
        for param in tqdm([(X, y, wnb, sklearn, i, test_size) for i in range(max_iter)], ncols=80)
    )

    return np.mean([r[0] for r in results]), np.mean([r[1] for r in results])
