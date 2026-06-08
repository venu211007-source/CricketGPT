import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load data
matches = pd.read_csv('matches.csv')

# Drop rows where winner is missing (no result matches)
matches = matches.dropna(subset=['winner'])

# Select features
# Add win rate as a feature
win_counts = matches['winner'].value_counts()
matches['team1_wins'] = matches['team1'].map(win_counts).fillna(0)
matches['team2_wins'] = matches['team2'].map(win_counts).fillna(0)

features = ['team1', 'team2', 'toss_winner', 'toss_decision', 'venue', 'team1_wins', 'team2_wins']
target = 'winner'

df = matches[features + [target]].copy()

# Encode all text columns to numbers
encoders = {}
for col in features + [target]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Split data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")