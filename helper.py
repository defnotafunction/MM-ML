import pandas as pd

m_teams = pd.read_csv('MTeams.csv')
m_teams.set_index('TeamID', inplace=True)
print(m_teams.head())
print(m_teams['1101'])