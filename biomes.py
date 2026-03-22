from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

breast_cancer = load_breast_cancer()

training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer.data, breast_cancer.target, test_size=.3, random_state=1)

scores = []
for i in np.arange(0, 10, 1):
    classifier = SVC(C=1)
    classifier.fit(training_data, training_labels)
    scores.append(classifier.score(validation_data, validation_labels))

x = np.arange(0, 10, 1)
plt.xlabel('Gamma')
plt.ylabel('Score')
plt.title('Gamma and Score', fontsize=20)
plt.legend()
plt.plot(x, scores)
plt.show()