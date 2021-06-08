#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:



mydata=pd.read_csv(r"https://raw.githubusercontent.com/98amanmaurya/Area-Type-Predictor/b301f1072a785cbfe4865eb74a514ac3638e93f7/project.csv")


# In[3]:


Xdata=mydata.iloc[:,:4]
Ydata=mydata.iloc[:,4:5]


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


Xtrain,Xtest,Ytrain,Ytest=train_test_split(Xdata,Ydata,test_size=.3,random_state=101)


# In[6]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.neighbors import KNeighborsClassifier


# In[7]:


TeacherG=GaussianNB()
TeacherB=BernoulliNB()
TeacherM=MultinomialNB()
TeacherK=KNeighborsClassifier()


# In[8]:


LearnerG=TeacherG.fit(Xtrain,Ytrain)
LearnerB=TeacherB.fit(Xtrain,Ytrain)
LearnerM=TeacherM.fit(Xtrain,Ytrain)
LearnerK=TeacherK.fit(Xtrain,Ytrain)


# In[9]:


YpB=LearnerB.predict(Xtest)
YpG=LearnerG.predict(Xtest)
YpM=LearnerM.predict(Xtest)
YpK=LearnerK.predict(Xtest)
Ya=Ytest


# In[10]:


from sklearn.metrics import accuracy_score


# In[11]:


accG=accuracy_score(Ya,YpG)*100
accB=accuracy_score(Ya,YpB)*100
accM=accuracy_score(Ya,YpM)*100
accK=accuracy_score(Ya,YpK)*100
acc=[accG,accM,accB,accK]
table=pd.DataFrame({"Acc":acc},index=["Gauss","Multi","Ber","Knn"])


# In[12]:


print(table)
print(LearnerK.predict([[9.0,87.0,45.0,110.0]]))
print(LearnerG.predict([[9.0,87.0,45.0,110.0]]))
print(LearnerM.predict([[9.0,87.0,45.0,110.0]]))
print(LearnerB.predict([[9.0,87.0,45.0,110.0]]))                


# In[ ]:




