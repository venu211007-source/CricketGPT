import streamlit as st
import pandas as pd

# Load data
matches = pd.read_csv('matches.csv')
deliveries = pd.read_csv('deliveries.csv')

st.title("🏏 CricketGPT — IPL Analytics Dashboard")
st.markdown("Explore IPL statistics and predict match outcomes.")

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