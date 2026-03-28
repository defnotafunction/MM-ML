from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from helper import *
import streamlit as st

@st.cache_resource
def train_forest():
    combined_data, combined_labels = get_vectorized_data()

    forest = RandomForestClassifier(random_state=1)
    forest.fit(combined_data, combined_labels)

    return forest

forest = train_forest()
team_names = m_teams['TeamName'].tolist()

st.title('March Madness Predictor')
team1 = st.selectbox("Select Team 1", team_names)
team2 = st.selectbox("Select Team 2", team_names)

if st.button("Predict Winner"):
    if not team1 or not team2:
        st.warning("Please enter both names")
    else:
        winner = team1 if predict_winner(team1, team2, forest) == 1 else team2
        st.success(f'The predicted winner is: {winner}')