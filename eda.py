import pandas as pd

deliveries = pd.read_csv('deliveries.csv')
matches = pd.read_csv('matches.csv')

print("Deliveries shape:", deliveries.shape)
print("Matches shape:", matches.shape)
print("\nMissing values in matches:")
print(matches.isnull().sum())

# Top 10 run scorers 
top_scorers = deliveries.groupby('batter')['batsman_runs'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Run Scorers:")
print(top_scorers)

# Top 10 wicket takers
wickets = deliveries[deliveries['is_wicket'] == 1]
top_bowlers = wickets.groupby('bowler')['is_wicket'].count().sort_values(ascending=False).head(10)
print("\nTop 10 Wicket Takers:")
print(top_bowlers)

import matplotlib.pyplot as plt

# Top 10 run scorers bar chart
top_scorers.plot(kind='bar', figsize=(10,5), color='steelblue')
plt.title('Top 10 IPL Run Scorers')
plt.xlabel('Player')
plt.ylabel('Total Runs')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top_scorers.png')
plt.show()

# Toss winner vs match winner
matches['toss_won_match'] = matches['toss_winner'] == matches['winner']
toss_win_pct = matches['toss_won_match'].value_counts(normalize=True) * 100
print("\nDoes winning toss help?")
print(toss_win_pct)

# Most successful teams
team_wins = matches['winner'].value_counts().head(10)
print("\nMost Wins by Team:")
print(team_wins)

# Plot it
team_wins.plot(kind='bar', figsize=(10,5), color='coral')
plt.title('Most IPL Wins by Team')
plt.xlabel('Team')
plt.ylabel('Number of Wins')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('team_wins.png')
plt.show()