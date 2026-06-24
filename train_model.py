from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

df = pd.read_csv(r"C:\Users\Dharani priya.U\Downloads\gallery\OneDrive\Student_Grade_Prediction_Dataset.csv")

X = df[['Study_Hours','Attendance_%',
        'Assignments_Score','Midterm_Score',
        'Final_Score']]

y = df['Grade']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully")