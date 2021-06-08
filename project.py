import pandas as pd
mydata=pd.read_csv(r"https://github.com/98amanmaurya/Area-Type-Predictor/blob/b301f1072a785cbfe4865eb74a514ac3638e93f7/project.csv")
Xdata=mydata.iloc[:,:4]
Ydata=mydata.iloc[:,4:5]
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(Xdata,Ydata,test_size=.30,random_state=101)
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
TeacherG=GaussianNB()
TeacherB=BernoulliNB()
TeacherM=MultinomialNB()
TeacherK=KNeighborsClassifier()
LearnerG=TeacherG.fit(Xtrain,Ytrain)
LearnerB=TeacherB.fit(Xtrain,Ytrain)
LearnerM=TeacherM.fit(Xtrain,Ytrain)
LearnerK=TeacherK.fit(Xtrain,Ytrain)
YpB=LearnerB.predict(Xtest)
YpG=LearnerG.predict(Xtest)
YpM=LearnerM.predict(Xtest)
YpK=LearnerK.predict(Xtest)
Ya=Ytest
from sklearn.metrics import accuracy_score
accG=accuracy_score(Ya,YpG)*100
accB=accuracy_score(Ya,YpB)*100
accM=accuracy_score(Ya,YpM)*100
accK=accuracy_score(Ya,YpK)*100
acc=[accG,accM,accB,accK]
table=pd.DataFrame({"Acc":acc},index=["Gauss","Multi","Ber","Knn"])
print(table)
print(LearnerK.predict([[9.0,87.0,45.0,110.0]]))
print(LearnerG.predict([[9.0,87.0,45.0,110.0]]))
print(LearnerM.predict([[9.0,87.0,45.0,110.0]]))
print(LearnerB.predict([[9.0,87.0,45.0,110.0]]))                