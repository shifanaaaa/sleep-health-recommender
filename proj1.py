import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

ml_df=df.copy()
ml_df['Sleep Quality Target']=ml_df['Quality of Sleep'].apply(lambda x:1 if x>=7 else 0)
feature=['Age','Sleep Duration','Physical Activity Level','Stress Level','Heart Rate']
target='Sleep Quality Target'
X=ml_df[feature]
y=ml_df[target]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("Preparing Data for Machine Learning")
print(f"Testing set size:{X_test.shape[0]} samples")

model=LogisticRegression()
model.fit(X_train,y_train)
print("Training and Evaluating the model")
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Model Accuracy:{accuracy:.2f}")
new_data=[[45,6.5,40,6,75]]
prediction=model.predict(new_data)
predicted_quality='Good Sleep' if prediction[0]==1 else 'Bad Sleep'
print(f"\nPrediction for a new person: They are likely to have '{predicted_quality}'")

df[['Systolic','Diastolic']]=df['Blood Pressure'].str.split('/',expand=True)
df['Systolic']=pd.to_numeric(df['Systolic'])
df['Diastolic']=pd.to_numeric(df['Diastolic'])
def health_rec(person_data):
    message=[]
    if person_data['Sleep Duration'] < 7:
        message.append("Looks like your sleep duration is on the lower side. Try to aim at least 7 hours for better focus and health.")
    if person_data['Physical Activity Level'] < 30:
        message.append("Looks like your activity level is on the low side. Try adding some movement to your day — even a short walk helps!")
    if person_data['Stress Level'] >= 7:
        message.append( "Looks like stress is piling up on you. Try to find some moments to relax and unwind — you deserve it!")
    if person_data['BMI Category'] == "Overweight":
        message.append( "Your BMI suggests you might be carrying a bit extra weight. Small changes in your daily habits can make a big difference — don’t rush, and take it one step at a time!")
    elif person_data['BMI Category'] == "Underweight":
        message.append("Your BMI is a bit lower than the healthy range. It might help to add some nutritious, energy-rich foods to your meals — your body needs fuel to feel its best!")
    systolic=person_data['Systolic']
    diastolic=person_data['Diastolic']
    if systolic>180 or diastolic>120:
        message.append("Hypertensive crisis.Seek Emergency care")
    elif systolic>=140 or diastolic>=90:
        message.append("High BP.Please consult a doctor")
    elif 130<=systolic<140 or 80<=diastolic<90:
        message.append( "You have high blood pressure. Consider making lifestyle changes such as healthier eating and increased activity, and consult a healthcare provider for personalized guidance.")
    elif 120<=systolic<130 and diastolic<80:
        message.append("Your blood pressure is elevated. Monitor it regularly and consider improving your diet and physical activity level to help bring it under control.")
    if person_data['Heart Rate']>100:
        message.append( "Hey, your heart rate seems a bit higher than usual. Try to take some relaxing breaks, keep an eye on it, and if it keeps up, chatting with a healthcare pro might be a good idea!")
    if person_data['Sleep Disorder']=="Sleep Apnea":
        message.append( "Looks like you might be having trouble breathing well during sleep, which can affect your energy and health. It’s a good idea to talk to a healthcare provider to explore solutions that can help you rest better.")
    elif person_data['Sleep Disorder']=="Insomnia":
        message.append( "Struggling to fall or stay asleep? Try creating a calming bedtime routine and limiting screen time before bed. If sleepless nights keep happening, consider reaching out to a sleep specialist for extra support.")
    if person_data['Daily Steps']<5000:
        message.append( "Your step count is quite low today. Try to move a little more — even a short walk can make a big difference!")
    return message
