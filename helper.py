import pandas as pd
import random
import numpy as np

# TEAM STATS
recent_seasons = [2022, 2023, 2024, 2025]

regular_stats = pd.read_csv("MRegularSeasonDetailedResults.csv")
tourney_stats = pd.read_csv("MNCAATourneyDetailedResults.csv")

stats_columns = [
    'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',
    'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF'
]
win_stats_columns = [f'W{s}' for s in stats_columns]
lost_stats_columns = [f'L{s}' for s in stats_columns]

# TEAM IDS
m_teams = pd.read_csv('MTeams.csv')
all_team_ids = m_teams['TeamID'].tolist()

m_teams.set_index('TeamID', inplace=True)

# HELPER FUNCTIONS
def get_team_from_id(team_id: int) -> str:
    """Gets team row from id"""
    return m_teams.loc[team_id]

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

def get_team_averages(team_name) -> pd.Series:
    """Gets the average stats of a team"""
    team_id = get_id_from_teamname(team_name)
    regular_avg = create_regular_averages_frame().loc[team_id]
    tourney_avg = create_tourney_averages_frame().loc[team_id].fillna(regular_avg)
    combined_avg = pd.concat([regular_avg, tourney_avg], axis=0)
    return combined_avg

def get_vectorized_data_from_teams(team1, team2):
    team1_avg, team2_avg = get_team_averages(team1), get_team_averages(team2)
    differences = [t1 - t2 for t1, t2 in zip(team1_avg.values, team2_avg.values)]
    return differences

def get_vectorized_data(seed=1, _type: str ='regular', seasons: list = recent_seasons):
    random.seed(seed)
    if _type == 'regular':
        regular_averages = create_regular_averages_frame(seasons)
        team_stats_dict = regular_averages.to_dict(orient='index')
        valid_team_ids = set(team_stats_dict.keys())
    elif _type == 'tourney':
        tourney_averages = create_tourney_averages_frame(seasons)
        team_stats_dict = tourney_averages.to_dict(orient='index')
        valid_team_ids = set(tourney_averages.index)
    else:
        combined = create_regular_averages_frame(seasons).join(
                                            create_tourney_averages_frame(seasons),
                                            how='inner',
                                            lsuffix='_reg',
                                            rsuffix='_tourney'
                                        )
        team_stats_dict = combined.to_dict(orient='index')
        valid_team_ids = set(combined.index)

    features = []
    labels = []

    all_winning_teams, all_losing_teams = regular_stats['WTeamID'], regular_stats['LTeamID']

    for w1, l1 in zip(all_winning_teams, all_losing_teams):
        if w1 not in valid_team_ids or l1 not in valid_team_ids:
            continue

        team1 = team_stats_dict[w1]
        team2 = team_stats_dict[l1]

        team1_vals = list(team1.values())
        team2_vals = list(team2.values())

        if any(np.isnan(team1_vals)) or any(np.isnan(team2_vals)):
            continue

        if random.random() > 0.5:
            differences = [team1_val - team2_val for team1_val, team2_val in zip(team1_vals, team2_vals)]
            label = 1  # 1 = win
        else:
            differences = [team2_val - team1_val for team1_val, team2_val in zip(team1_vals, team2_vals)]
            label = 0

        labels.append(label)
        features.append(differences)

    return np.array(features), np.array(labels)

def get_mode_of_predictions(features, *models):
    predictions = [model.predict([features])[0] for model in models]
    return max(set(predictions), key=predictions.count)

def predict_winner(team1, team2, *models):
    features = get_vectorized_data_from_teams(team1, team2)
    prediction = get_mode_of_predictions(features, *models)
    return prediction