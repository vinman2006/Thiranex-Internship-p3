import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.naive_bayes import (
    MultinomialNB
)

from sklearn.pipeline import (
    Pipeline
)

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)

import joblib
import matplotlib.pyplot as plt

data = pd.read_csv(
    "dataset.csv"
)

X = data["email"]
y = data["label"]

X_train,\
X_test,\
y_train,\
y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Pipeline([

(
"vector",

TfidfVectorizer(
stop_words="english"
)

),

(
"classifier",

MultinomialNB()
)

])

model.fit(
X_train,
y_train
)

pred = model.predict(
X_test
)

print(
"\nAccuracy:"
)

print(
accuracy_score(
y_test,
pred
)
)

print(
"\nReport:\n"
)

print(
classification_report(
y_test,
pred
)
)

cm = confusion_matrix(
y_test,
pred
)

plt.figure(
figsize=(5,4)
)

plt.imshow(cm)

plt.title(
"Confusion Matrix"
)

plt.colorbar()

plt.xlabel(
"Predicted"
)

plt.ylabel(
"Actual"
)

plt.show()

joblib.dump(
model,
"model.pkl"
)

print(
"\nModel Saved"
)
