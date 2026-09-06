import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

data={ 
      "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 2, 6],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 90, 58, 85],
    "Pass": [0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
}

df=pd.DataFrame(data)

X=df[["Hours_Studied","Attendance"]]
y = df["Pass"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model=DecisionTreeClassifier(criterion="gini",max_depth=3,min_samples_split=5,min_samples_leaf=2,random_state=42)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))