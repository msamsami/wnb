from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from wnb import GaussianWNB


# Load the breast cancer wisconsin dataset
breast_cancer = load_breast_cancer()
X = breast_cancer["data"]
y = breast_cancer["target"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0
)

# Train and score sklearn GaussianNB classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("sklearn | GaussianNB >> score >>", gnb.score(X_test, y_test))

# Train and score wnb GaussianWNB classifier
gwnb = GaussianWNB(max_iter=20, step_size=0.01)
gwnb.fit(X_train, y_train)
print("wnb | GaussianWNB >> score >>", gwnb.score(X_test, y_test))
