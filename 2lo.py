import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import(accuracy_score,confusion_matrix,classification_report)

X = np.array([
    [2, 50],
    [3, 55],
    [4, 60],
    [5, 65],
    [6, 70],
    [7, 75],
    [8, 80],
    [9, 85],
    [10, 90],
    [11, 95]
])

y = np.array([
    0, 0, 0, 0, 1,
    1, 1, 1, 1, 1
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

model=LogisticRegression()
model.fit(X_train_scaled,y_train)
y_pred=model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print(cm)

print(classification_report(y_test, y_pred))