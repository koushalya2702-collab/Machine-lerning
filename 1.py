from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
X=[[1],[2],[3],[4],[5],[6],[7],[8]]
y=[0,0,0,0,1,1,1,1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
print("classification report:")
print(classification_report(y_test,y_pred))