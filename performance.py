import pandas as pd
import numpy as np

import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Step 1: Load the Data
data = [
    {"Quiz Title": "Human Physiology (15)", "Score": 108, "Accuracy": 90, "Correct": 27, "Incorrect": 3, "Duration": 15,
     "Date": "2025-01-17"},
    {"Quiz Title": "Human Physiology PYQ", "Score": 92, "Accuracy": 100, "Correct": 23, "Incorrect": 0, "Duration": 15,
     "Date": "2025-01-17"},
    {"Quiz Title": "Reproduction", "Score": 40, "Accuracy": 38, "Correct": 10, "Incorrect": 16, "Duration": 15,
     "Date": "2025-01-15"},
    # Add more rows as necessary
]

df = pd.DataFrame(data)

# Step 2: Preprocess Data
df['Date'] = pd.to_datetime(df['Date'])
df['Days_Since_First_Quiz'] = (df['Date'] - df['Date'].min()).dt.days

# Select features for ML
features = ['Score', 'Accuracy', 'Correct', 'Incorrect', 'Duration', 'Days_Since_First_Quiz']
X = df[features]
y = df['Score']  # Target variable for regression

# Step 3: Apply Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
df['Cluster'] = clusters

# Add labels for clusters
df['Performance_Label'] = df['Cluster'].map({
    0: "Consistently Strong",
    1: "Average Performer",
    2: "Needs Improvement"
})

# Step 4: Train Regression Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train, y_train)
with open('regressor.pkl', "wb") as clf_file:
    pickle.dump(regressor, clf_file)
# Predict and Evaluate
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error (Regression): {mse}")

# Step 5: Train Classification Model for Recommendations
df['Weak_Topic'] = (df['Accuracy'] < 50).astype(int)  # Define weak topic based on accuracy
X_class = df[features]
y_class = df['Weak_Topic']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
classifier = LogisticRegression(random_state=42)
classifier.fit(X_train_c, y_train_c)
with open("classifier.pkl", "wb") as clf_file:
    pickle.dump(classifier, clf_file)
# Classification Report
y_pred_c = classifier.predict(X_test_c)
print("Classification Report for Weak Topic Detection:")
print(classification_report(y_test_c, y_pred_c))

# Step 6: Generate Recommendations
recommendations = df[df['Weak_Topic'] == 1]['Quiz Title'].unique()

# Step 7: Highlight Strengths and Weaknesses
for _, row in df.iterrows():
    if row['Weak_Topic'] == 1:
        print(f"Weakness: Focus on {row['Quiz Title']} (Accuracy: {row['Accuracy']}%)")
    else:
        print(f"Strength: Strong performance in {row['Quiz Title']} (Accuracy: {row['Accuracy']}%)")

print("\nOverall Recommendations:")
print("Recommended Topics to Focus On:", recommendations)

# Step 8: Summary Insights
strengths = df[df['Weak_Topic'] == 0]['Quiz Title'].unique()
weaknesses = df[df['Weak_Topic'] == 1]['Quiz Title'].unique()
print("\nPerformance Summary:")
print(f"Strengths: {', '.join(strengths)}")
print(f"Weaknesses: {', '.join(weaknesses)}")

with open("classifier.pkl", "rb") as clf_file:
    loaded_classifier = pickle.load(clf_file)


# Function to take user input
# Function to take user input
def get_user_input():
    print("\nPlease enter the details of your quiz:")
    try:
        Quiztitle = eval(input("enter the title of quiz:"))
        score = float(input("Enter the Score: "))
        accuracy = float(input("Enter the Accuracy (%): "))
        correct = int(input("Enter the number of Correct Answers: "))
        incorrect = int(input("Enter the number of Incorrect Answers: "))
        duration = int(input("Enter the Quiz Duration (minutes): "))
        days_since_first_quiz = int(input("Enter the Days Since First Quiz: "))

        # Validate input ranges (optional, based on dataset characteristics)
        if accuracy < 0 or accuracy > 100:
            print("Accuracy should be between 0 and 100.")
            return get_user_input()
        if score < 0 or correct < 0 or incorrect < 0 or duration <= 0:
            print("Please enter valid positive values.")
            return get_user_input()

        # Return input as a DataFrame
        return pd.DataFrame([{
            "Quiztitle": Quiztitle,
            "Score": score,
            "Accuracy": accuracy,
            "Correct": correct,
            "Incorrect": incorrect,
            "Duration": duration,
            "Days_Since_First_Quiz": days_since_first_quiz
        }])
    except ValueError:
        print("Invalid input! Please enter valid numeric values.")
        return get_user_input()


# Get user input
user_input = get_user_input()

# Ensure user input aligns with model training features
expected_features = ['Score', 'Accuracy', 'Correct', 'Incorrect', 'Duration', 'Days_Since_First_Quiz']
user_input = user_input[expected_features]  # Align columns if necessary

# Use the classification model to predict if it's a weak topic
try:
    classification_prediction = loaded_classifier.predict(user_input)
    print("\nModel Predictions:")
    if classification_prediction[0] == 1:
        print("Weak Topic Detected (Classification): Yes")
        print("Recommendation: Focus more on this topic to improve.")
    else:
        print("Weak Topic Detected (Classification): No")
        print("Keep up the good work!")
except Exception as e:
    print(f"Error during prediction: {e}")

# Summary for the user
print("\nSummary of Your Input:")
for col in user_input.columns:
    print(f"{col}: {user_input.iloc[0][col]}")
