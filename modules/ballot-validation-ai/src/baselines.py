"""
BASELINE để đối chứng, huấn luyện trên cùng 8 đặc trưng cấu trúc:
  majority   : luôn đoán lớp phổ biến nhất (mốc tối thiểu)
  naive_bayes: Gaussian Naive Bayes
  knn        : k-NN (k=5)
"""
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def train_baselines(F_tr, y_tr):
    models = {
        "majority": DummyClassifier(strategy="most_frequent").fit(F_tr, y_tr),
        "naive_bayes": GaussianNB().fit(F_tr, y_tr),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)).fit(F_tr, y_tr),
    }
    return models
