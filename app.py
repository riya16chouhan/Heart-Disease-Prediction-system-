import streamlit as st
import pandas as pd
import joblib 

model=joblib.load('KNN_heart.pkl')
scaler=joblib.load('scaler (5).pkl')
expected_columns= joblib.load('columns.pkl')


st.title("Heart Disease Prediction App")
st.info("Provide your clinical report details to know the prediction.")
age=st.slider('Age',18,100,30)
sex=st.selectbox('Sex',['Male','Female'])
chest_pain=st.selectbox('Chest Pain Type',["ATA","NAP","ASY","TA"])
resting_bp=st.number_input("Resting Blood Pressure (in mm Hg)",80,200,120)
cholesterol=st.number_input("Cholesterol (in mg/dl)",100,600,200)
fasting_bs=st.selectbox("Fasting Blood Sugar > 120 mg/dl",[1,0])
resting_ecg=st.selectbox("Resting ECG",["Normal","ST","LVH"])
max_hr=st.slider("Max Heart Rate",60,220,150)
exercise_angina=st.selectbox("Exercise Induced Angina",["Yes","No"])
oldpeak=st.slider("Oldpeak-ST depression",0.0,6.0,1.0)
st_slope=st.selectbox("ST Slope",["Up","Flat","Down"])


if st.button("Predict"):
    raw_input={
        'age':age,
        'RestingBP':resting_bp,
        'Cholesterol':cholesterol,
        'FastingBS':fasting_bs,
        'MaxHR':max_hr,
        'Oldpeak':oldpeak,
        'Sex_' + sex:1 ,
        'ChestPainType_' + chest_pain:1,
        'RestingECG_' + resting_ecg:1,
        'ST_Slope_' + st_slope:1,
        'ExerciseAngina_' + exercise_angina:1
   }
    
    input_df=pd.DataFrame([raw_input])
    
    
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col]=0
            
    input_df=input_df[expected_columns]        
    
    scaled_input=scaler.transform(input_df)
    probability = model.predict_proba(scaled_input)[0][1]
    probability_percent = probability * 100

    if probability < 0.5:
        st.success("No chances of heart attack")
    elif probability < 0.8:
        st.warning("Low chances of heart attack")
    else:
        st.error("High chances of heart attack")

    st.write(f"Probability of heart attack: {probability_percent:.1f}%")
    st.progress(float(probability))

    st.markdown("---")
    st.info("This project is for educational purposes only.")
