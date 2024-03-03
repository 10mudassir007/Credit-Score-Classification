import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)

df = pd.read_csv('bank.csv')

#Categorical Features
categorical_features = ['Gender','Education','Marital Status','Home Ownership']
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(df[categorical_features])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_features))

#Combining orignal with categorical
df = df.join(encoded_df)
df = df.drop(categorical_features,axis=1)

#Explanatory Data Analysis
#Pie chart of all the labels
value_counts = df[['Credit Score']].value_counts()
high = value_counts[0]
average = value_counts[1]
low = value_counts[2]

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
l = ['High', 'Average', 'Low']
s = [high,average,low]
ax.pie(s, labels = l,autopct='%1.2f%%')


#Histograms of all the features
df.hist()


#Variablity in the features
df.plot(kind ='density',subplots = True, layout =(14,3),sharex = False)

#?
sns.pairplot(df,hue='Credit Score')


#Splitting data into X and y
y = np.array(df[['Credit Score']])
X = np.array(df.drop(['Credit Score'],axis=1))

#Relation between features
plt.xlabel("Features")
plt.ylabel("Credit Score")
plt.scatter(df['Income'].values.reshape(-1,1),df['Number of Children'].values.reshape(-1,1),color='b')

plt.show()

#Scaling Data
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Splitting Data into training and test sets
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

#Intializing, Training and making predictions on the model
model = KNeighborsClassifier(n_neighbors=2)
model.fit(x_train,y_train)
predictions = model.predict(x_test)



#Metrics
accuracy = accuracy_score(y_test,predictions)
report = classification_report(y_test,predictions)
cm = confusion_matrix(y_test,predictions)


print(f'-----------------------------------------------Classification Report--------------------------------------------------------\n{report}')
print('Accuracy Score',accuracy)
print('Confusion matrix\n',cm)


#Confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


