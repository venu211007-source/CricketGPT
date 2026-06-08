# Model Accuracy Improvement Log

## Goal
Predict the winner of an IPL match using pre-match information.

## Iteration 1 — Random Forest (Baseline)
Features: team1, team2, toss_winner, toss_decision, venue  
Model: Random Forest (100 trees)  
Accuracy: 47.71%  
Observation: Basic features not sufficient. Toss and venue alone don't determine outcomes.

## Iteration 2 — Random Forest + Historical Win Counts
Features:team1, team2, toss_winner, toss_decision, venue, team1_wins, team2_wins  
Model: Random Forest (100 trees)  
Accuracy: 46.79%  
Observation: Adding win counts slightly hurt accuracy — possibly due to teams that no longer exist in IPL skewing historical data.

## Iteration 3 — XGBoost + Historical Win Counts
Features: team1, team2, toss_winner, toss_decision, venue, team1_wins, team2_wins  
Model: XGBoost (100 estimators)  
Accuracy: 49.08%  
Observation: XGBoost outperforms Random Forest on this dataset. Best result so far.

## Key Insight
Pre-match prediction in cricket is inherently difficult. Published research shows accuracy rarely exceeds 65% using only pre-match features. In-match features like current run rate, wickets fallen, and required run rate would significantly improve prediction accuracy.

## Next Steps
- Merge ball-by-ball data to create in-match features
- Predict win probability at each over
- Explore deep learning approaches