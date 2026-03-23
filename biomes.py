from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
from helper import *
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

regular_features, regular_labels = get_vectorized_data()
r_training_data, r_testing_data, r_training_labels, r_testing_labels = train_test_split(regular_features, regular_labels, test_size=0.3, random_state=1)

tourney_features, tourney_labels = get_vectorized_data(_type='tourney')
t_training_data, t_testing_data, t_training_labels, t_testing_labels = train_test_split(tourney_features, tourney_labels, test_size=0.3, random_state=1)

t_train_scaled = scaler.fit_transform(t_training_data)
t_test_scaled = scaler.transform(t_testing_data)

#forest = RandomForestClassifier(random_state=1)
#forest.fit(r_training_data, r_training_labels)
#print(forest.score(t_testing_data, t_testing_labels))


#regression = LogisticRegression()
#regression.fit(t_training_data, t_training_labels)
#print(regression.score(t_testing_data, t_testing_labels))

# RandomForest peak = 0.6063811034725894

svc = SVC(C=.37)
svc.fit(t_train_scaled, t_training_labels)
print(svc.score(t_test_scaled, t_testing_labels))