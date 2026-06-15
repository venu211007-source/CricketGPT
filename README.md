# CricketGPT

This project is an attempt to apply Machine Learning and Data Analytics to cricket. The goal is to build a system that can analyze historical cricket data, generate insights about players and teams, predict match outcomes, and eventually simulate entire tournaments.

I have always been interested in cricket, especially leagues like the IPL, so I wanted to work on a project that combines that interest with AI and Machine Learning.

## Current Status

Completed:
- Repository setup
- Dataset collection
- Exploratory Data Analysis (EDA)
- Win probability estimation
- Match winner prediction model
- Interactive Streamlit dashboard
- Tournament simulation



## Dataset

The project uses historical cricket match data, including match-level and ball-by-ball information. The dataset will be used for exploratory analysis, feature engineering, and training prediction models.

## Technologies

Some of the tools and libraries I plan to use are:

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- XGBoost
- Streamlit

## Current Status

The project is currently in its transitional and implementational stage.

Completed:
- Repository setup
- Dataset collection
- Explore and clean the data
- Perform exploratory data analysis
- Create features for prediction models

Next Steps:
- Train and evaluate machine learning models

## File Descriptions

eda.py
Performs exploratory data analysis on the IPL dataset. Includes top run scorers, top wicket takers, toss impact analysis, and team win counts. Generates and saves bar charts as PNG files.

model.py (completed after multiple testing)
Will contain the match winner prediction model. Uses match-level features like team names, venue, and toss decision to predict the winning team using machine learning.

app.py (currently working on)
Streamlit-based interactive dashboard to visualize player and team statistics and run live match predictions.

## Note 
The model.py is used only for testing purposes to find the accuracy and improve it while app.py is used for the frontend dashboard(Streamlit) purposes with no link between model.py

## Why this project?

Most beginner ML projects focus on standard datasets such as sentiment analysis or spam detection. Through this project, I want to work with a larger real-world dataset and gain experience in data analysis, feature engineering, predictive modeling, and deployment.

The long-term goal is to create a platform that can provide meaningful cricket insights and predictions using data-driven methods.

## Result so far

Pre-match prediction accuracy of ~49%, consistent with published research showing cricket match outcomes are highly stochastic. In-match prediction using live ball-by-ball features is planned as the next improvement. Currently the dashboard is being test and devloped.I have built the live win probability feature after testing its accuaracy and its performance over a period test cases.Now currently tournament simulations process is completed.