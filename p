import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report,
confusion_matrix
import matplotlib.pyplot as plt
iris = load_iris()
# Usar apenas 2 features para v i s u a l i z a o (se quiser trocar, me
avise)
X = iris.data[:, :2]
3
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# D i v i s o treino/teste
X_train, X_test, y_train, y_test = train_test_split(
X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
clf = Perceptron(max_iter=1000, eta0=0.01, random_state=42)
clf.fit(X_train, y_train)
# P r e d i o
y_pred = clf.predict(X_test)
print(" A c u r c i a :", accuracy_score(y_test, y_pred))
print("\nRelat rio de C l a s s i f i c a o :\n", classification_report(
y_test, y_pred))
print("Matriz de C o n f u s o :\n", confusion_matrix(y_test, y_pred))
def plot_decision_boundary(clf, X, y, title):
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
np.arange(x_min, x_max, 0.02),
np.arange(y_min, y_max, 0.02)
)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 5))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor=’k’)
plt.xlabel("Feature 1 (padronizada)")
plt.ylabel("Feature 2 (padronizada)")
plt.title(title)
plt.show()
plot_decision_boundary(clf, X_scaled, y, "Perceptron - Fronteira de
D e c i s o ")
