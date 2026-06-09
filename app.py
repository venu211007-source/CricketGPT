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
page = st.sidebar.selectbox("Choose a page", ["Overview", "Player Stats", "Team Stats", "Match Prediction"])

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
