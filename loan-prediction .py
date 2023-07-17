#!/usr/bin/env python
# coding: utf-8

# # Loan Eligibility Prediction 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


loan_data = pd.read_csv('train.csv')


# In[7]:


loan_data.head()


# In[8]:


loan_data.info()


# In[9]:


loan_data.shape


# In[10]:


loan_data.describe()


# In[11]:


#the relation between credit history and loan status


# In[12]:


pd.crosstab(loan_data['Credit_History'], loan_data['Loan_Status'], margins=True)


# In[13]:


loan_data.boxplot(column='ApplicantIncome')


# In[14]:


loan_data['ApplicantIncome'].hist(bins=20)


# In[15]:


loan_data['CoapplicantIncome'].hist(bins=20)


# In[16]:


loan_data.boxplot(column='ApplicantIncome', by='Education')


# In[17]:


loan_data.boxplot(column='LoanAmount')


# In[18]:


loan_data['LoanAmount'].hist(bins=20)


# In[19]:


loan_data['LoanAmount_log']=np.log(loan_data['LoanAmount'])


# In[20]:


#Taking the logarithm can help to normalize the data and reduce the impact of extreme values,
#making it easier to work with or visualize.


# In[21]:


loan_data['LoanAmount_log'].hist(bins=20)


# In[22]:


#null data 


# In[23]:


loan_data.isnull().sum()


# In[24]:


#filling the missing value with mode 


# In[25]:


loan_data['Gender'].fillna(loan_data['Gender'].mode()[0],inplace=True)


# In[26]:


loan_data['Married'].fillna(loan_data['Married'].mode()[0],inplace=True)


# In[27]:


loan_data['Dependents'].fillna(loan_data['Dependents'].mode()[0],inplace=True)


# In[28]:


loan_data['Self_Employed'].fillna(loan_data['Self_Employed'].mode()[0],inplace=True)


# In[29]:


loan_data['Loan_Amount_Term'].fillna(loan_data['Loan_Amount_Term'].mode()[0],inplace=True)


# In[30]:


loan_data['Credit_History'].fillna(loan_data['Credit_History'].mode()[0],inplace=True)


# In[31]:


#filling the missing value with mean


# In[32]:


loan_data.LoanAmount= loan_data.LoanAmount.fillna(loan_data.LoanAmount.mean())
loan_data.LoanAmount_log= loan_data.LoanAmount_log.fillna(loan_data.LoanAmount_log.mean())


# In[33]:


loan_data.isnull().sum()


# In[34]:


loan_data['TotalIncome'] = loan_data['ApplicantIncome'] + loan_data['CoapplicantIncome']


# In[35]:


loan_data['TotalIncome_log'] = np.log(loan_data['TotalIncome']) 


# In[36]:


loan_data['TotalIncome_log'].hist(bins=20)


# In[37]:


loan_data.head()


# In[38]:


#spliting the data into two input(X) and output(y)


# In[39]:


X= loan_data.iloc[:,np.r_[1:5,9:11,13:15]].values


# In[40]:


y= loan_data.iloc[:,12].values


# In[41]:


X


# In[42]:


y


# In[43]:


#Split the dataset into training and testing sets:


# In[44]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[45]:


print(X_train)


# In[46]:


print(X_test)


# In[47]:


print(y_test)


# In[48]:


#transforming categorical variables into numerical labels


# In[49]:


from sklearn.preprocessing import LabelEncoder


# In[50]:


labelencoder_X = LabelEncoder()


# In[51]:


for i in range(0,5):
    X_train[:,i] = labelencoder_X.fit_transform(X_train[:,i])


# In[52]:


X_train[:,7] = labelencoder_X.fit_transform(X_train[:,7])


# In[53]:


X_train


# In[54]:


labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)


# In[55]:


y_train


# In[56]:


for i in range(0,5):
    X_test[:,i] = labelencoder_X.fit_transform(X_test[:,i]) 


# In[57]:


X_test[:,7] = labelencoder_X.fit_transform(X_test[:,7])


# In[58]:


labelencoder_y = LabelEncoder()
y_test = labelencoder_y.fit_transform(y_test)


# In[59]:


X_test


# In[60]:


#normalization 
#Normalization is important because it brings all features to a similar scale,
#preventing one feature from dominating or biasing the learning algorithm over others.
#It ensures that each feature contributes proportionately to the learning process.


# In[61]:


from sklearn.preprocessing import StandardScaler 
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)


# In[62]:


y_test


# In[63]:


# train the model using decision tree algorithm


# In[64]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X_train,y_train)


# In[65]:


y_pred= model.predict(X_test)
y_pred


# In[66]:


from sklearn import metrics
print('The accuracy of decision tree is: ', metrics.accuracy_score(y_pred,y_test))
print('The precision of decision tree is: ', metrics.precision_score(y_pred,y_test))
print('The recall of decision tree is: ', metrics.recall_score(y_pred,y_test))
print('The F1 Score of decision tree is: ', metrics.f1_score(y_pred,y_test))


# In[67]:


# train the model using Navie Bayes algorithm


# In[68]:


from sklearn.naive_bayes import GaussianNB
modelnb = GaussianNB()
modelnb.fit(X_train,y_train)


# In[69]:


y_pred = modelnb.predict(X_test)


# In[70]:


y_pred


# In[71]:


print('The accuracy of naive bayes is: ', metrics.accuracy_score(y_pred,y_test))
print('The precision of naive bayes is: ', metrics.precision_score(y_pred,y_test))
print('The recall of naive bayes is: ', metrics.recall_score(y_pred,y_test))
print('The F1 Score of naive bayes is: ', metrics.f1_score(y_pred,y_test))


# In[72]:


#train the model using Random Forest algorithm 


# In[73]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,y_train)


# In[74]:


y_pred = model.predict(X_test)
y_pred


# In[75]:


print('The accuracy of random forest  is: ', metrics.accuracy_score(y_pred,y_test))
print('The precision of random forest is: ', metrics.precision_score(y_pred,y_test))
print('The recall of random forest is: ', metrics.recall_score(y_pred,y_test))
print('The F1 Score of random forest is: ', metrics.f1_score(y_pred,y_test))


# In[76]:


#trian the model using Linear Regression algorithm


# In[77]:


from sklearn.linear_model import LogisticRegression
modellr = LogisticRegression()
modellr.fit(X_train, y_train)


# In[78]:


y_pred = modellr.predict(X_test)
y_pred


# In[79]:


print('The accuracy of logistic regression  is: ', metrics.accuracy_score(y_pred,y_test))
print('The precision of logistic regression is: ', metrics.precision_score(y_pred,y_test))
print('The recall of logistic regression is: ', metrics.recall_score(y_pred,y_test))
print('The F1 Score of logistic regression is: ', metrics.f1_score(y_pred,y_test))


# In[80]:


#train the model using SVM algorithm


# In[81]:


from sklearn.svm import SVC
modelsvm = SVC()
modelsvm.fit(X_train, y_train)


# In[82]:


y_pred = modelsvm.predict(X_test)
y_pred


# In[83]:


print('The accuracy of SVM  is: ', metrics.accuracy_score(y_pred,y_test))
print('The precision of SVM is: ', metrics.precision_score(y_pred,y_test))
print('The recall of SVM is: ', metrics.recall_score(y_pred,y_test))
print('The F1 Score of SVM is: ', metrics.f1_score(y_pred,y_test))


# In[86]:


from sklearn.preprocessing import LabelEncoder

# Create label encoder and fit it on the target variable y_train
label_encoder = LabelEncoder()
label_encoder.fit(y_train)

# Encode the target variable y_train and y_test
y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Create reverse label encoder to convert predictions back to original labels
reverse_label_encoder = LabelEncoder()
reverse_label_encoder.fit(label_encoder.classes_)


# In[99]:


# Function to get user input
def get_user_input():
    gender = input("Gender (Male/Female): ")
    married = input("Married (Yes/No): ")
    dependents = input("Dependents (0/1/2/3+): ")
    education = input("Education (Graduate/Not Graduate): ")
    self_employed = input("Self Employed (Yes/No): ")
    applicant_income = float(input("Applicant Income: "))
    loan_amount = float(input("Loan Amount: "))
    loan_amount_term = float(input("Loan Amount Term (in months): "))

    # Preprocess user input
    user_data = np.array([[gender, married, dependents, education, self_employed, applicant_income, loan_amount, loan_amount_term]])

    # Encode categorical variables
    for i in range(0, 5):
        if user_data[0, i] in label_encoder.classes_:
            user_data[:, i] = label_encoder.transform(user_data[:, i])
        else:
            # Handle unknown labels by assigning a unique integer value
            user_data[:, i] = len(label_encoder.classes_)

    # Standardize numerical variables
    user_data = ss.transform(user_data)

    return user_data


# In[135]:


# Get user input
user_data = get_user_input()

# Make prediction
prediction = model.predict(user_data)

# Convert prediction back to original label
prediction = reverse_label_encoder.inverse_transform(prediction)

if prediction == 1:
    print("Prediction: Eligible")
else:
    print("Prediction: Not Eligible")


# In[134]:


import pickle

# Save the trained SVM model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
    


# In[ ]:




