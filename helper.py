import pandas as pd
import random
import numpy as np

# TEAM STATS
recent_seasons = [2022, 2023, 2024, 2025]

regular_stats = pd.read_csv("MMData/MRegularSeasonDetailedResults.csv")
tourney_stats = pd.read_csv("MMData/MNCAATourneyDetailedResults.csv")

stats_columns = [
    'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',
    'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF'
]
win_stats_columns = [f'W{s}' for s in stats_columns]
lost_stats_columns = [f'L{s}' for s in stats_columns]

# TEAM IDS
m_teams = pd.read_csv('MMData/MTeams.csv')
all_team_ids = m_teams['TeamID'].tolist()

m_teams.set_index('TeamID', inplace=True)

# HELPER FUNCTIONS
def get_team_from_id(team_id: int) -> str:
    """Gets team row from id"""
    return m_teams.loc[team_id]['TeamName']

def get_id_from_teamname(team_name: str) -> int:
    """Gets ID from name of team"""
    row = m_teams[m_teams['TeamName'] == team_name]
    return row.index[0]

def get_recent_team_stats(dataframe: pd.DataFrame, recent_years: int):
    """Filters the database, removing all rows that contains seasons before 2020."""
    return dataframe[dataframe['Season'].isin(recent_years)]

def create_regular_averages_frame(seasons=recent_seasons) -> pd.DataFrame:
    """Uses the recent regular season statistics to make a table with the average stats for each team."""
    teams_averages = {}
    regular_stats_recent = get_recent_team_stats(regular_stats, seasons)

    for team_id in all_team_ids:
        team_wins = regular_stats_recent[regular_stats_recent['WTeamID'] == team_id]
        team_losses = regular_stats_recent[regular_stats_recent['LTeamID'] == team_id]
        combined = pd.concat([
            team_wins[win_stats_columns],
            team_losses[lost_stats_columns].rename(columns=lambda x: x.replace('L', 'W'))
        ])
        team_avg_columns = combined.mean()
        teams_averages[team_id] = {stat[1:]:avg for stat, avg in team_avg_columns.items()}

    return pd.DataFrame(teams_averages).T

def create_tourney_averages_frame(seasons=recent_seasons) -> pd.DataFrame:
    """Uses the recent tourney season statistics to make a table with the average stats for each team."""
    teams_averages = {}
    tourney_stats_recent = get_recent_team_stats(tourney_stats, seasons)

    for team_id in all_team_ids:
        team_wins = tourney_stats_recent[tourney_stats_recent['WTeamID'] == team_id]
        team_losses = tourney_stats_recent[tourney_stats_recent['LTeamID'] == team_id]
        combined = pd.concat([
            team_wins[win_stats_columns],
            team_losses[lost_stats_columns].rename(columns=lambda x: x.replace('L', 'W'))
        ])
        team_avg_columns = combined.mean()
        teams_averages[team_id] = {stat[1:]: avg for stat, avg in team_avg_columns.items()}

    return pd.DataFrame(teams_averages).T

REGULAR_AVG = create_regular_averages_frame()
TOURNEY_AVG = create_tourney_averages_frame()

def get_team_averages(team_name):
    team_id = get_id_from_teamname(team_name)

    regular_avg = REGULAR_AVG.loc[team_id]
    tourney_avg = TOURNEY_AVG.loc[team_id].fillna(regular_avg)

    combined = pd.concat([
        regular_avg.add_suffix('_reg'),
        tourney_avg.add_suffix('_tourney')
    ])

    return combined

def get_vectorized_data_from_teams(team1, team2):
    team1_avg, team2_avg = get_team_averages(team1), get_team_averages(team2)
    differences = [t1 - t2 for t1, t2 in zip(team1_avg.values, team2_avg.values)]
    return differences


def get_vectorized_data(seed=1):
    random.seed(seed)

    features = []
    labels = []

    for w, l in zip(regular_stats['WTeamID'].to_list(), regular_stats['LTeamID'].to_list()):
        if w not in REGULAR_AVG.index or l not in REGULAR_AVG.index:
            continue
        
        w = get_team_from_id(w)
        l = get_team_from_id(l)

        if random.random() > 0.5:
            features.append(get_vectorized_data_from_teams(w, l))
            labels.append(1)
        else:
            features.append(get_vectorized_data_from_teams(l, w))
            labels.append(0)

    return np.array(features), np.array(labels)

def get_mode_of_predictions(features, *models):
    predictions = [model.predict([features])[0] for model in models]
    return max(set(predictions), key=predictions.count)

def predict_winner(team1, team2, *models):
    features = get_vectorized_data_from_teams(team1, team2)
    prediction = get_mode_of_predictions(features, *models)
    return prediction