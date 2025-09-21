import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("logistics_Regression_md.pkl")
df = pd.read_csv(r"C:\Users\USER\Downloads\Loan_default.csv")

st.title("Welcome to T-Loan")

st.header("Use the model to predict whether the borrower will default on the loan.")

edu_opt = ["High School", "Bachelor", "Master", "PhD"]
em_opt = ["Full-time", "Part-time", "Self-employed", "Unemployed"]
loan_pur = ["Auto", "Business", "Education", "Home", "Other"]
mar_opt =  ["Divorced", "Married", "Single"]

def oh_encoding(cat, options, prefix):
    one_hot = [1 if cat == option else 0 for option in options]
    #column = [f"{prefix}_{option}" for option in options]
    #encoded_df = pd.DataFrame([one_hot], columns=column)
    return one_hot

def ordinal(cat, options, prefix):
    return (int(options.index(cat)))

def label(cat):
    label = 1 if cat else 0
    return label



with st.form("loan_form"):
    st.header("Basic information")
    Age = st.number_input("Enter the age of the individual", min_value=18, max_value=60, step=1)
    Income = st.number_input("Enter the annual income: ", min_value=0)
    LoanAmount = st.number_input("Enter the amount of money borrowed by the individual: ", min_value=0)
    CreditScore = st.number_input("Enter the current credit score: ", min_value=0)
    MonthsEmployed = st.number_input("For how many months has the individual been employed now: ", min_value=0)
    NumCreditLines = st.number_input("How many open credit accounts do the indidual has: ", min_value=0)
    InterestRate = st.number_input("What is the annual loan interest for the individual: ")
    LoanTerm = st.number_input("What is the duration of the loan in(Month): ", min_value=1)
    DTIRatio = st.number_input("What is the total debit relative to income: ")
    Education = st.radio("Education level", edu_opt)
    HasMortgage = st.checkbox("Has Mortgage")
    HasDependents = st.checkbox("Has Dependents")
    HasCosigner = st.checkbox("Has Co-Signer")
    LoanPurpose = st.radio("Loan Purpose", loan_pur)
    EmploymentType = st.radio("Employment Type", em_opt)
    MaritalStatus = st.radio("Marital Status", mar_opt)
    
    submitted = st.form_submit_button()

if submitted:
    st.success("Data Collected Successfully!")

    education = ordinal(Education, edu_opt, "Education")

    hasMortgage = label(HasMortgage)

    hasDependents =label(HasDependents)

    hasCosigner =label(HasCosigner)

    loanPurpose = oh_encoding(LoanPurpose, loan_pur, "LoanPurpose")

    employmentType = oh_encoding(EmploymentType, em_opt, "EmploymentType")

    maritalStatus = oh_encoding(MaritalStatus, mar_opt, "MaritalStatus")

    Data = [Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio, education, hasMortgage, hasDependents, hasCosigner] + loanPurpose + employmentType + maritalStatus

    column_names = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
       'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'Education',
       'HasMortgage', 'HasDependents', 'HasCoSigner', 'LoanPurpose_Auto',
       'LoanPurpose_Business', 'LoanPurpose_Education', 'LoanPurpose_Home',
       'LoanPurpose_Other', 'EmploymentType_Full-time',
       'EmploymentType_Part-time', 'EmploymentType_Self-employed',
       'EmploymentType_Unemployed', 'MaritalStatus_Divorced',
       'MaritalStatus_Married', 'MaritalStatus_Single']

    st.dataframe(pd.DataFrame([Data], columns=column_names))

    input_data = np.array(Data)
    reshape_input = input_data.reshape(1, -1)
    prediction = model.predict(reshape_input)

    Result = "The Client will not Default" if prediction == 0 else "The Client will Default. Loan out at your own risk."

    st.write(Result)

    #st.dataframe(encoded_df2)
    #st.dataframe(pd.concat([encoded_df,encoded_df2], axis=1))


   