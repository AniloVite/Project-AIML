import pickle
import pandas as pd
import warnings
#For the program to not show the warnings:
warnings.filterwarnings('ignore')
try:
    with open('best_model.pkl','rb') as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    print("Failed to load scaler.pkl")
#Writing all the features which are important to predict loan default
features = ['Annual_Income', 'Loan_Amount', 'Loan_Term_Months', 'Credit_Score',
            'Existing_Loans_Count', 'Debt_to_Income_Ratio', 'Savings_Balance']
#User inputt area
print("Loan default predictor dataset")
print("Input Data:\n")
A = {}
for i in features:
    a = input(f"Enter {i}:")
    A[i] = a
#Converting list into dataset:
df = pd.DataFrame([A])
#Scaling given data
df = scaler.transform(df)
# Predicting Loan Default
pred = model.predict(df)
if pred[0] == 1:
    print("Predicted Loan Default:High Risk")
else:
    print("Predicted Loan Default:Low Risk")