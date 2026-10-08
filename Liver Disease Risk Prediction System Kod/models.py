# models.py
# Karaciğer hastalığı tahmini için sınıflandırıcı ve topluluk öğrenmesi modelleri

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# 1) Klasik sınıflandırıcılar
def get_classifiers():
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    return classifiers

# 2) Topluluk öğrenmesi yöntemleri
def get_ensemble_models(base_estimator=None):
    if base_estimator is None:
        base_estimator = DecisionTreeClassifier(random_state=42)
    ensembles = {
        "Bagging": BaggingClassifier(estimator=base_estimator, n_estimators=50, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42)
    }
    return ensembles

# 3) Yapay Sinir Ağı (ANN) - sklearn MLPClassifier
def get_ann(input_dim):
    # hidden_layer_sizes: (64,32) → iki gizli katman
    # max_iter: 100+ → epoch yerine iterasyon sayısı
    ann = MLPClassifier(hidden_layer_sizes=(64, 32),
                        activation='relu',
                        solver='adam',
                        max_iter=200,   # 100+ iterasyon
                        random_state=42)
    return ann