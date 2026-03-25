from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from helper import *
from sklearn.preprocessing import StandardScaler
import streamlit as st

print(get_team_averages('Duke'))  # Sanity check

scaler = StandardScaler()


# Regular and Tourney datasets combined
combined_data, combined_labels = get_vectorized_data(_type='both')
combined_training, combined_testing, combined_training_labels, combined_test_labels = train_test_split(combined_data, combined_labels, test_size=0.3, random_state=1)
c_train_scaled = scaler.fit_transform(combined_training)
c_test_scaled = scaler.transform(combined_testing)


forest = RandomForestClassifier(random_state=1)
forest.fit(combined_training, combined_training_labels)
#print(forest.score(combined_testing, combined_test_labels))


regression = LogisticRegression()
regression.fit(combined_training, combined_training_labels)
#print(regression.score(combined_testing, combined_test_labels))


svc = SVC(C=.37)
svc.fit(c_train_scaled, combined_training_labels)
#print(svc.score(c_test_scaled, combined_test_labels))

team_names = m_teams['TeamName'].tolist()

st.title('March Madness Predictor')
team1 = st.selectbox("Select Team 1", team_names)
team2 = st.selectbox("Select Team 2", team_names)

if st.button("Predict Winner"):
    if not team1 or not team2:
        st.warning("Please enter both names")
    else:
        winner = team1 if predict_winner(team1, team2, svc, regression, forest) == 1 else team2
        st.success(f'The predicted winner is: {winner}')