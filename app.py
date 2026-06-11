import streamlit as st
import pandas as pd

# Load data
matches = pd.read_csv('matches.csv')
deliveries = pd.read_csv('deliveries.csv')

st.title("🏏 CricketGPT — IPL Analytics Dashboard")
st.markdown("Explore IPL statistics and predict match outcomes.")

st.caption("Data source: Kaggle IPL Dataset (2008–2024). Stats may vary slightly from official records.")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Overview", "Player Stats", "Team Stats", "Match Prediction", "Win Probability"])

if page == "Overview":
    st.header("📊 IPL Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Matches", len(matches))
    col2.metric("Total Deliveries", len(deliveries))
    col3.metric("Seasons", matches['season'].nunique())

    st.subheader("Top 10 Run Scorers")
    top_scorers = deliveries.groupby('batter')['batsman_runs'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_scorers)

    st.subheader("Top 10 Wicket Takers")
    wickets = deliveries[deliveries['is_wicket'] == 1]
    top_bowlers = wickets.groupby('bowler')['is_wicket'].count().sort_values(ascending=False).head(10)
    st.bar_chart(top_bowlers)

elif page == "Player Stats":
    st.header("Player Stats")
    player = st.selectbox("Select Player", sorted(deliveries['batter'].unique()))
    player_runs = deliveries[deliveries['batter'] == player]['batsman_runs'].sum()
    player_matches = deliveries[deliveries['batter'] == player]['match_id'].nunique()
    player_avg = round(player_runs / player_matches, 2)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Runs", player_runs)
    col2.metric("Matches Played", player_matches)
    col3.metric("Runs per Match", player_avg)

elif page == "Team Stats":
    st.header("Team Stats")
    team = st.selectbox("Select Team", sorted(matches['team1'].unique()))
    team_wins = matches[matches['winner'] == team].shape[0]
    team_matches = matches[(matches['team1'] == team) | (matches['team2'] == team)].shape[0]
    win_pct = round((team_wins / team_matches) * 100, 2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Matches", team_matches)
    col2.metric("Total Wins", team_wins)
    col3.metric("Win %", f"{win_pct}%")
elif page == "Match Prediction":
    st.header("Match Prediction")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import numpy as np

    df = matches.dropna(subset=['winner'])
    features = ['team1', 'team2', 'toss_winner', 'toss_decision', 'venue']
    target = 'winner'
    df = df[features + [target]].copy()

    encoders = {}
    for col in features + [target]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df[features]
    y = df[target]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    teams = sorted(matches['team1'].unique())
    venues = sorted(matches['venue'].unique())

    team1 = st.selectbox("Team 1", teams)
    team2 = st.selectbox("Team 2", [t for t in teams if t != team1])
    toss_winner = st.selectbox("Toss Winner", [team1, team2])
    toss_decision = st.selectbox("Toss Decision", ["bat", "field"])
    venue = st.selectbox("Venue", venues)

    if st.button("Predict Winner"):
        input_data = pd.DataFrame([[team1, team2, toss_winner, toss_decision, venue]], columns=features)
        for col in features:
            le = encoders[col]
            input_data[col] = input_data[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        prediction = model.predict(input_data)[0]
        winner = encoders[target].inverse_transform([prediction])[0]
        st.success(f"Predicted Winner: {winner}")
elif page == "Win Probability":
        st.header("Win Probability Estimator")
        st.markdown("Estimate the winning probability of the chasing team based on current match situation.")

        target = st.number_input("Target Runs", min_value=1, max_value=300, value=180)
        runs_scored = st.number_input("Runs Scored So Far", min_value=0, max_value=300, value=50)
        wickets_fallen = st.number_input("Wickets Fallen", min_value=0, max_value=10, value=2)
        overs_completed = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, value=6.0, step=0.1)

        runs_required = target - runs_scored
        overs_remaining = 20 - overs_completed
        rrr = round(runs_required / overs_remaining, 2) if overs_remaining > 0 else 999
        crr = round(runs_scored / overs_completed, 2) if overs_completed > 0 else 0

        st.subheader("Match Situation")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Runs Required", runs_required)
        col2.metric("Overs Remaining", round(overs_remaining, 1))
        col3.metric("Required Run Rate", rrr)
        col4.metric("Current Run Rate", crr)

         # Simple probability formula based on RRR vs CRR and wickets
        wickets_factor = (10 - wickets_fallen) / 10
        rr_factor = crr / rrr if rrr > 0 else 1
        win_prob = round(min(max((rr_factor * wickets_factor) * 100, 0), 100), 1)

        st.subheader("Win Probability")
        st.progress(int(win_prob))
        st.metric("Chasing Team Win Probability", f"{win_prob}%")

        if win_prob >= 90:
          st.success("Chasing team is likely to win")    
        elif win_prob >= 60:
          st.success("Chasing team is in a strong position!") 
        elif win_prob >= 40:
          st.warning("Match is evenly poised.")
        elif win_prob <=20:
            st.error("Defending team is likely to win")  
        else:
          st.error("Defending team is in a strong position!")