import joblib
import pandas as pd

pipeline = joblib.load('titanic_pipeline.pkl')

# Test passenger — 3rd class male, age 22, alone
passenger = pd.DataFrame([{
    'Pclass': 3, 'Sex': 0, 'Age': 22.0, 'Fare': 7.25,
    'Embarked': 0, 'FamilySize': 1, 'IsAlone': 1,
    'Has_Cabin': 0, 'Title': 0,
}])

pred = pipeline.predict(passenger)[0]
prob = pipeline.predict_proba(passenger)[0][1]
print(f"Survived: {'Yes' if pred == 1 else 'No'}")
print(f"Probability: {prob:.1%}")



# predict.py:

# loads saved trained pipeline
# creates new passenger input
# predicts survival
# outputs probability

# Important concepts:

# predict() → final class
# predict_proba() → confidence/probability
# DataFrame format must match training features

# Main concept:

# A trained ML model becomes useful when it can make 
# predictions on completely new unseen real-world data 
# outside the notebook environment