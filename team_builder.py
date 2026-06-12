import streamlit as st
import pandas as pd
import json

st.title("IPL Team Builder")
st.markdown("Enter your custom teams and players for the tournament simulation.")

# Number of teams
num_teams = st.number_input("Number of Teams", min_value=2, max_value=10, value=10, step=1)

teams = {}

for i in range(int(num_teams)):
    st.subheader(f"Team {i+1}")
    team_name = st.text_input(f"Team Name", key=f"team_{i}")
    players_input = st.text_area(f"Enter Players (one per line)", key=f"players_{i}", height=150)
    
    if team_name:
        players = [p.strip() for p in players_input.split('\n') if p.strip()]
        teams[team_name] = players

if st.button("Save Teams"):
    if len(teams) < 2:
        st.error("Enter at least 2 teams.")
    else:
        incomplete = [t for t, p in teams.items() if len(p) < 11]
        if incomplete:
            st.warning(f"These teams have less than 11 players: {', '.join(incomplete)}")
        
        with open('custom_teams.json', 'w') as f:
            json.dump(teams, f)
        
        st.success("Teams saved successfully!")
        st.subheader("Your Teams")
        for team, players in teams.items():
            st.write(f"**{team}** ({len(players)} players)")
            st.write(", ".join(players))