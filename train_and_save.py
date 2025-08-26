# train_and_save.py
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
# при желании: from xgboost import XGBClassifier; from lightgbm import LGBMClassifier

RANDOM_STATE = 42

# 0) загрузка/получение данных
# df = pd.read_csv("heart.csv")  # пример, подставь свой путь
# ожидаемые столбцы:
# ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS',
#  'RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope','HeartDisease']
# Ниже оставлю заглушку: бросай сюда свой df
#
# raise SystemExit("Загрузи свой DataFrame в переменную df и закомментируй эту строку.")
df = pd.read_csv('aux/heart.csv')
y = df["HeartDisease"]
X = df.drop(columns=["HeartDisease"])

# 1) группы колонок
ordinal_cols = ['Sex', 'ExerciseAngina', 'ST_Slope', 'FastingBS']
onehot_cols  = ['RestingECG', 'ChestPainType']
num_mean_cols       = ['Age']
num_zero_mean_cols  = ['Cholesterol']  # 0 трактуем как пропуск
num_other_num_cols  = ['RestingBP', 'MaxHR', 'Oldpeak']

# 2) подпроцессы
ordinal_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

ohe_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),  # без drop — стабильно на фолдах
])

num_mean_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="mean")),
    ("scaler", MinMaxScaler()),
])

num_zero_mean_pipe = Pipeline([
    ("imp0", SimpleImputer(missing_values=0, strategy="mean")),
    ("scaler", MinMaxScaler()),
])

num_other_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler()),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("ord", ordinal_pipe,       ordinal_cols),
        ("ohe", ohe_pipe,           onehot_cols),
        ("age", num_mean_pipe,      num_mean_cols),
        ("chol", num_zero_mean_pipe,num_zero_mean_cols),
        ("num", num_other_pipe,     num_other_num_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)
# (по желанию) чтобы downstream видеть имена:
# preprocessor.set_output(transform="pandas")

# 3) твоя «лучшая» модель (подставь из Optuna). Ниже — разумный дефолт:
best_model = RandomForestClassifier(
    n_estimators=600, max_depth=None, min_samples_leaf=1,
    random_state=RANDOM_STATE, n_jobs=-1
)
# пример замены:
# best_model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4,
#                            subsample=0.9, colsample_bytree=0.9,
#                            objective="binary:logistic", eval_metric="logloss",
#                            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist")

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", best_model),
])

# 4) обучаем на всём датасете (или train — на твой выбор)
pipe.fit(X, y)

# 5) сохраняем
joblib.dump(pipe, "heart_pipeline.pkl")
print("✅ Saved to heart_pipeline.pkl")
