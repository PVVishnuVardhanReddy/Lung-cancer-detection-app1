import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


import pandas as pd
import numpy as np







df=pd.read_csv('C:\\app\\cancer patient data sets.csv')

df.drop('Patient Id', inplace=True, axis=1)


df['Level'] = df['Level'].replace({'Low': 0, 'Medium': 1,'High':2})


st.title("Lung Cancer Diagnosis")
df.drop(['index','Air Pollution','OccuPational Hazards','Genetic Risk','Balanced Diet','Obesity','Passive Smoker','Weight Loss','Frequent Cold','Snoring','Coughing of Blood'], inplace=True, axis=1)
df.drop(['Gender','Dry Cough','Clubbing of Finger Nails','chronic Lung Disease','Swallowing Difficulty','Alcohol use'],inplace=True,axis=1)

y1= df.iloc[:,-1]
x1= df.iloc[:,0:7]
x_train1,x_test1,y_train1,y_test1= train_test_split(x1,y1,random_state=1,train_size=0.2)
import xgboost as xgb
from sklearn.metrics import accuracy_score
dtrain1 = xgb.DMatrix(x_train1, label=y_train1)
dtest1 = xgb.DMatrix(x_test1, label=y_test1)
params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 3,
    'eta': 0.1,
    'eval_metric': 'merror'
}
num_rounds = 10
model = xgb.train(params, dtrain1, num_rounds)
y_pred11 = model.predict(dtest1)
accuracy11= accuracy_score(y_test1,y_pred11)
def user_report():
    options1 = ['Male', 'Female']
    options = ['True', 'False']
    age = st.slider('Age',0,100,25)
    gender = st.radio("Select Gender", options1)
    
    dustallergy = st.slider('Dust Allergy',0,10,0)
   
    smoking = st.slider('smoking',0,10,0)
   
    chestpain = st.slider('chest pain',0,10,0)
    
    fatigue = st.slider('fatigue',0,10,0)
    
    shortnessofbreath = st.slider('Shortness of breath',0,10,0)
    wheezing = st.slider('wheezing',0,10,0)
    
    user_report = {
            'Age': age,
            'Dust Allergy': dustallergy,
            'Smoking': smoking,
            'Chest Pain': chestpain,
            'Fatigue': fatigue,
            'Shortness of Breath': shortnessofbreath,
            'Wheezing': wheezing,
            
    }

    report_data = pd.DataFrame(user_report, index=[0])
    return report_data
user_data=user_report()

st.write(user_data)
dtest11 = xgb.DMatrix(user_data, label=[0,1,2])
res=model.predict(dtest11)
st.write(res)


st.subheader("with accuracy:")
st.write(accuracy11*100)
st.subheader("Your Report:")
if res[0]==2:
    st.warning('Your are suffering from Lung Cancer')
else:
    st.success('You are not suffering from Lung Cancer')
    st.balloons()
