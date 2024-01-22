from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from util.util import metric_eval, load_data
import xgboost as xgb

# 获取数据的维度
num_people = 137  # 数据中的人数
num_features = 137  # 特征数量
feature_dim = 116  # 特征形状

num_folds = 5
n_components = 20

param = {
    'n_estimators': 300,
    'max_depth': 9,
    'learning_rate': 0.01,
}


# Create a StratifiedKFold object
kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=126357)

X, y = load_data()

for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # pca = PCA(n_components=n_components)

    X_train = X_train.reshape(-1, num_features * feature_dim)
    X_test = X_test.reshape(-1, num_features * feature_dim)

    # 创建XGBoost模型
    model = xgb.XGBClassifier(**param)
    # 训练模型
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc, sen, spe, ppv, npv = metric_eval(y_test, y_pred)
    print(f'acc: {acc:.4f}\nsen: {sen:.4f}\nspe: {spe:.4f}\n'
          f'ppv: {ppv:.4f}\nnpv:{npv:.4f}')
