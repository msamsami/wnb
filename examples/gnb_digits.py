import warnings

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from wnb import Distribution as D
from wnb import GeneralNB

warnings.filterwarnings("ignore")


# Load the digits dataset
digits = load_digits()
X = digits["data"]
y = digits["target"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train and score sklearn GaussianNB classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("sklearn | GaussianNB >> score >>", gnb.score(X_test, y_test))

# Train and score wnb GeneralNB classifier with Poisson likelihoods
gnb = GeneralNB(distributions=[D.POISSON] * X.shape[1])
gnb.fit(X_train, y_train)
print("wnb | GeneralNB >> score >>", gnb.score(X_test, y_test))
