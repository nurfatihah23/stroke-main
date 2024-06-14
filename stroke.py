import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

# Set the title and description of the app
st.title('Stroke Prediction')
st.markdown("""
    Welcome to the Stroke Prediction App. This application helps to predict the likelihood of a stroke based on various health parameters.
    Fill out the details below to get your prediction.
""")

# Load the dataset
dataset = pd.read_csv("C:/Users/MateBook/Desktop/New folder/sem 7/BCI3333 MLA/archive_15/stroke_prediction_dataset.csv")

# Drop unnecessary columns
dataset = dataset.drop(columns=['Marital Status', 'Work Type', 'Symptoms', 'Residence Type', 'Patient ID', 'Patient Name', 'Cholesterol Levels', 'Blood Pressure Levels'])

# User input features
st.sidebar.header('User Input Features')
st.sidebar.markdown("**Please provide the following details:**")

# Collect user input features
age = st.sidebar.number_input("Age: ", 0.00, 200.00, 20.00, 1.0)
gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
hypertension = st.sidebar.selectbox('Hypertension', ('0: No Hypertension', '1: Hypertension'))
heart_disease = st.sidebar.selectbox('Heart Disease', ('0: No Heart Disease', '1: Heart Disease'))
glucose_level = st.sidebar.slider('Glucose Level',  0.00, 200.00, 100.00, 0.500)
bmi = st.sidebar.number_input("BMI: ", 0.00, 100.00, 15.00, 0.01)
smoking_status = st.sidebar.selectbox('Smoking Status', ('Non-smoker', 'Formerly Smoked', 'Currently Smokes'))
alcohol_intake = st.sidebar.selectbox('Alcohol Intake', ('Never', 'Rarely', 'Frequent Drinker', 'Social Drinker'))
physical_activity = st.sidebar.selectbox('Physical Activity', ('Low', 'Moderate', 'High'))
stroke_history = st.sidebar.selectbox('Stroke History', ('0: No Stroke History', '1: Has Stroke History'))
family_stroke = st.sidebar.selectbox('Family History of Stroke', ('No', 'Yes'))
diatery_habits = st.sidebar.selectbox('Diatery Habits', ('Vegan', 'Paleo', 'Pescatarian', 'Gluten-Free', 'Vegetarian', 'Non-Vegetarian'))
stress_level = st.sidebar.slider('Stress Level',  0.00, 20.00, 1.00, 0.10)

# Create a dictionary of user input data
data = {
    'Age': age,
    'Gender': gender,
    'Hypertension': hypertension, 
    'Heart Disease': heart_disease,
    'Average Glucose Level': glucose_level,
    'Body Mass Index (BMI)': bmi, 
    'Smoking Status': smoking_status,
    'Alcohol Intake': alcohol_intake,
    'Physical Activity': physical_activity,
    'Stroke History': stroke_history,
    'Family History of Stroke': family_stroke,
    'Diatery Habits': diatery_habits,
    'Stress Levels': stress_level
}

# Convert the dictionary to a DataFrame
user_input_df = pd.DataFrame(data, index=[0])

# Predict button
if st.sidebar.button('Predict'):
    # Encode and scale the features
    categorical_columns = user_input_df.select_dtypes(include=['object']).columns

    # Apply OneHotEncoder to categorical columns
    ct = ColumnTransformer(
        transformers=[
            ("encoder", OneHotEncoder(), categorical_columns)
        ],
        remainder='passthrough'
    )

    # Fit and transform the dataset
    dataset_transformed = ct.fit_transform(user_input_df)

    # Convert the transformed dataset back to a DataFrame
    dataset_transformed = pd.DataFrame(dataset_transformed)

    # Load the saved model
    load_model = pickle.load(open('C:/Users/MateBook/Desktop/New folder/sem 7/BCI3333 MLA/archive_15/stroke_svm.pkl', 'rb'))

    # Apply model to make predictions
    prediction = load_model.predict(dataset_transformed)

    # Display prediction
    st.subheader('Prediction')
    status = {0: 'No-Stroke', 1: 'Stroke'}
    st.markdown(f'**The diagnosis is: {status[prediction[0]]}**')

    # Visualize user input
    st.subheader('User Input Parameters')
    st.write(user_input_df)

    # Adding an informative image
    st.image('https://www.cdc.gov/stroke/images/stroke-types.jpg', caption='Types of Stroke', use_column_width=True)

# Footer
st.markdown("""
    ---
    **Note**: This prediction is based on a machine learning model and should not be considered as a medical diagnosis. Please consult a healthcare professional for medical advice.
""")
