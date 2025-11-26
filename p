import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)

spam_words = np.random.normal(loc=15, scale=3, size=(50, 1))
spam_links = np.random.normal(loc=7, scale=2, size=(50, 1))
spam = np.hstack((spam_words, spam_links))

nao_words = np.random.normal(loc=5, scale=2, size=(50, 1))
nao_links = np.random.normal(loc=1, scale=1, size=(50, 1))
nao_spam = np.hstack((nao_words, nao_links))

X = np.vstack((spam, nao_spam))
y = np.array([1]*50 + [0]*50)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

clf = Perceptron(max_iter=1000, eta0=0.01, random_state=42)
clf.fit(X_train, y_train)

print("Acurácia:", clf.score(X_test, y_test))

def plot_decision_boundary(clf, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")
    plt.xlabel("Quantidade de palavras (padronizada)")
    plt.ylabel("Quantidade de links (padronizada)")
    plt.title("Classificação Spam x Não Spam - Perceptron")
    plt.show()

plot_decision_boundary(clf, X_scaled, y)

