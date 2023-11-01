# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: karnan k
RegisterNumber:  212222230062
*/
import chardet
file = '/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd 
data = pd.read_csv("/content/spam.csv",encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()



x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)


from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)


y_pred = svc.predict(x_test)
y_pred


from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### data.head()
![image](https://github.com/karnankasinathan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118787064/f8466aa9-ed51-4b2f-a27a-efe25d055842)

### data.info()
![image](https://github.com/karnankasinathan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118787064/6d64a069-64c8-4c70-b096-d0a9d6f6a623)

### data.isnull().sum()
![image](https://github.com/karnankasinathan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118787064/6ca1038f-f023-4a05-855c-8d29f6ddfc6a)

### Y_prediction value
![image](https://github.com/karnankasinathan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118787064/83b9522d-5397-45bb-b58b-36efc3975060)

### Accuracy value
![image](https://github.com/karnankasinathan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118787064/ffc219fc-84e4-410e-8001-fdd10cec30e3)

### Result
![image](https://github.com/karnankasinathan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118787064/8ded628d-8151-4357-b120-e61b7e8bb7e3)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
