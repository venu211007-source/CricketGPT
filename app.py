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
page = st.sidebar.selectbox("Choose a page", ["Overview", "Player Stats", "Team Stats", "Match Prediction", "Win Probability", "Tournament Simulation"])

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
    
    st.markdown("**Overs Bowled**")
    col_ov1, col_ov2 = st.columns(2)
    with col_ov1:
        overs = st.number_input("Completed Overs", min_value=0, max_value=20, value=6, step=1)
    with col_ov2:
        # If they hit 20 completed overs, max balls forced to 0
        max_balls = 0 if overs == 20 else 5 
        balls = st.number_input("Additional Balls", min_value=0, max_value=max_balls, value=0, step=1)

    # --- MATH CORRECTIONS FOR CRICKET LOGIC ---
    # Convert everything to total balls to keep the math 100% accurate
    total_balls_bowled = (overs * 6) + balls
    total_match_balls = 120
    balls_remaining = total_match_balls - total_balls_bowled
    
    # Convert total balls back into a true cricket decimal representation for display (e.g. 6.2)
    overs_completed = overs + (balls / 10)
    overs_remaining = (balls_remaining // 6) + ((balls_remaining % 6) / 10)
    
    # Decimal overs used strictly for Run Rate mathematical division
    overs_completed_div = total_balls_bowled / 6
    overs_remaining_div = balls_remaining / 6

    runs_required = target - runs_scored
    
    # Calculate Run Rates accurately using exact fractional overs
    rrr = round(runs_required / overs_remaining_div, 2) if overs_remaining_div > 0 else (0.0 if runs_required <= 0 else 999)
    crr = round(runs_scored / overs_completed_div, 2) if overs_completed_div > 0 else 0.0

    st.subheader("Match Situation")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Runs Required", runs_required)
    col2.metric("Overs Remaining", f"{int(overs_remaining)}.{int((overs_remaining%1)*10)}")
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

elif page == "Tournament Simulation":
    st.header("IPL Tournament Simulation")
    st.markdown("Simulates a full IPL season using historical match data with realistic outcomes.")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import itertools
    import random
    import numpy as np

    # Train prediction model
    df_m = matches.dropna(subset=['winner'])
    features = ['team1', 'team2', 'toss_winner', 'toss_decision', 'venue']
    target = 'winner'
    df_enc = df_m[features + [target]].copy()

    encoders = {}
    for col in features + [target]:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        encoders[col] = le

    sim_model = RandomForestClassifier(n_estimators=100, random_state=42)
    sim_model.fit(df_enc[features], df_enc[target])

    # Current IPL teams
    import json
    import os

    # Load custom teams if available, else use default IPL teams
    if os.path.exists('custom_teams.json'):
       with open('custom_teams.json', 'r') as f:
        custom_teams = json.load(f)
       ipl_teams = list(custom_teams.keys())
       st.success("Using your custom teams!")
    else:
        ipl_teams = [
            "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bengaluru",
            "Kolkata Knight Riders", "Delhi Capitals", "Rajasthan Royals",
            "Punjab Kings", "Sunrisers Hyderabad", "Gujarat Titans", "Lucknow Super Giants"
        ]
    st.info("Using default IPL teams. Build custom teams in Team Builder.")

    venues = list(matches['venue'].unique())

    def predict_winner(t1, t2):
        venue = random.choice(venues)
        toss_w = random.choice([t1, t2])
        toss_d = random.choice(["bat", "field"])
        row = pd.DataFrame([[t1, t2, toss_w, toss_d, venue]], columns=features)
        for col in features:
            le = encoders[col]
            row[col] = row[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
        pred = sim_model.predict(row)[0]
        model_winner = encoders[target].inverse_transform([pred])[0]

        # Add 35% upset probability to keep results realistic
        if random.random() < 0.25:
            return random.choice([t1, t2])
        return model_winner if model_winner in [t1, t2] else random.choice([t1, t2])

    if st.button("Run Tournament Simulation"):

        # League stage — each team plays every other team once (9 matches per team)
        points = {team: 0 for team in ipl_teams}
        league_results = []

        base_fixtures = list(itertools.combinations(ipl_teams, 2))
        extra_fixtures = []
        match_count = {team: 9 for team in ipl_teams}

        attempts = 0
        while min(match_count.values()) < 14 and attempts < 1000:
            attempts += 1
            available = [t for t in ipl_teams if match_count[t] < 14]
            if len(available) < 2:
                break
            t1, t2 = random.sample(available, 2)
            extra_fixtures.append((t1, t2))
            match_count[t1] += 1
            match_count[t2] += 1

        fixtures = base_fixtures + extra_fixtures
        for t1, t2 in fixtures:
            winner = predict_winner(t1, t2)
            loser = t2 if winner == t1 else t1
            points[winner] += 2
            league_results.append({"Team 1": t1, "Team 2": t2, "Winner": winner})

        # Points table
        points_df = pd.DataFrame(list(points.items()), columns=["Team", "Points"])
        points_df = points_df.sort_values("Points", ascending=False).reset_index(drop=True)
        points_df.index += 1

        st.subheader("League Stage — Points Table")
        st.dataframe(points_df)

        top4 = points_df["Team"].head(4).tolist()
        st.info(f"Top 4 qualified: {', '.join(top4)}")
# Playoffs
        st.subheader("Playoffs")

        # Qualifier 1 — 1st vs 2nd (winner goes to final)
        q1_winner = predict_winner(top4[0], top4[1])
        q1_loser = top4[1] if q1_winner == top4[0] else top4[0]
        st.write(f"**Qualifier 1:** {top4[0]} vs {top4[1]} → **{q1_winner}** advances to Final")

        # Eliminator — 3rd vs 4th (loser eliminated)
        elim_winner = predict_winner(top4[2], top4[3])
        elim_loser = top4[3] if elim_winner == top4[2] else top4[2]
        st.write(f"**Eliminator:** {top4[2]} vs {top4[3]} → **{elim_winner}** survives, {elim_loser} eliminated")

        # Qualifier 2 — Q1 loser vs Eliminator winner (winner goes to final)
        q2_winner = predict_winner(q1_loser, elim_winner)
        q2_loser = elim_winner if q2_winner == q1_loser else q1_loser
        st.write(f"**Qualifier 2:** {q1_loser} vs {elim_winner} → **{q2_winner}** advances to Final, {q2_loser} eliminated")

        st.subheader("Final")
        st.write(f"**{q1_winner}** vs **{q2_winner}**")        