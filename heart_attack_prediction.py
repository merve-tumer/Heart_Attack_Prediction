

import joblib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
# !pip install catboost
# !pip install lightgbm
# !pip install xgboost
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import train_test_split, cross_validate

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Veri setinin okunması
df = pd.read_csv("data\heart_attack_prediction_dataset.csv")
df.head()


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

check_df(df)

##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

df["Systolic_Blood_Pres"] = df["Blood Pressure"].apply(lambda x: x.split("/")[0]).astype(int)
df["Diastolic_Blood_Pres"] = df["Blood Pressure"].apply(lambda x: x.split("/")[1]).astype(int)
df.drop("Blood Pressure", axis=1, inplace=True)




def grab_col_names(dataframe, cat_th=4, car_th=30):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    #num_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

drop_list = ['Patient ID', 'Country', 'Continent', 'Hemisphere', 'Obesity', 'Medication Use']
df.drop(drop_list, axis=1, inplace=True)
cat_cols, num_cols, cat_but_car = grab_col_names(df)

##################################
# FEATURE ENGINEERING
##################################

df.loc[(df["Age"] >= 18) & (df["Age"] < 40), "NEW_AGE_CAT"] = "young"
df.loc[(df["Age"] >= 40) & (df["Age"] < 65), "NEW_AGE_CAT"] = "mature"
df.loc[(df["Age"] >= 65), "NEW_AGE_CAT"] = "senior"

# df.groupby("NEW_AGE_CAT")["Heart Attack Risk"].mean()


df.loc[(df['Sex'] == 'Male') & (df["Age"] >= 18) & (df["Age"] < 40), 'NEW_SEX_CAT'] = 'youngmale'

df.loc[(df['Sex'] == 'Male') & (df["Age"] >= 40) & (df["Age"] < 65), 'NEW_SEX_CAT'] = 'maturemale'

df.loc[(df['Sex'] == 'Male') & (df["Age"] >= 65), 'NEW_SEX_CAT'] = 'seniormale'

df.loc[(df['Sex'] == 'Female') & (df["Age"] >= 18) & (df["Age"] < 40), 'NEW_SEX_CAT'] = 'youngfemale'

df.loc[(df['Sex'] == 'Female') & (df["Age"] >= 40) & (df["Age"] < 65), 'NEW_SEX_CAT'] = 'maturefemale'

df.loc[(df['Sex'] == 'Female') & (df["Age"] >= 65), 'NEW_SEX_CAT'] = 'seniorfemale'

#df.groupby("NEW_SEX_CAT")["Heart Attack Risk"].mean()

## Mapping for diet
diet_mapping = {"Unhealthy" : 1, "Average": 2, "Healthy": 3}
df["Diet"] = df["Diet"].map(diet_mapping)

df['New_Income_Diet'] = df['Diet'] * df['Income']

df["risk_factor"] = df['Family History'] + df['Previous Heart Problems'] + df['Alcohol Consumption'] + df['Smoking']


# New feature
df["Choles_Trig"] = df["Cholesterol"] / df["Triglycerides"]


# BMI 18,5 aşağısı underweight, 18.5 ile 24.9 arası normal, 24.9 ile 29.9 arası Overweight ve 30 üstü obez
df['NEW_BMI'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=["Underweight", "Healthy", "Overweight", "Obese"])

# Yaş ve beden kitle indeksini bir arada düşünerek kategorik değişken oluşturma
df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 18) & (df["Age"] < 40)), "NEW_AGE_BMI_NOM"] = "underweight_young"
df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 40) & (df["Age"] < 65)), "NEW_AGE_BMI_NOM"] = "underweight_mature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 65), "NEW_AGE_BMI_NOM"] = "underweight_senior"

df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & ((df["Age"] >= 18) & (df["Age"] < 40)), "NEW_AGE_BMI_NOM"] = "healthy_young"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & ((df["Age"] >= 40) & (df["Age"] < 65)), "NEW_AGE_BMI_NOM"] = "healthy_mature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 65), "NEW_AGE_BMI_NOM"] = "healthy_senior"

df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["Age"] >= 18) & (df["Age"] < 40)), "NEW_AGE_BMI_NOM"] = "overweight_young"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["Age"] >= 40) & (df["Age"] < 65)), "NEW_AGE_BMI_NOM"] = "overweight_mature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 65), "NEW_AGE_BMI_NOM"] = "overweight_senior"

df.loc[(df["BMI"] >= 30) & ((df["Age"] >= 18) & (df["Age"] < 40)), "NEW_AGE_BMI_NOM"] = "obese_young"
df.loc[(df["BMI"] >= 30) & ((df["Age"] >= 40) & (df["Age"] < 65)), "NEW_AGE_BMI_NOM"] = "obese_mature"
df.loc[(df["BMI"] >= 30) & (df["Age"] >= 65), "NEW_AGE_BMI_NOM"] = "obese_senior"


# BMI Dİabete Relation

df.loc[(df["BMI"] < 18.5) & (df["Diabetes"] == 1), "NEW_BMI_DIABET"] = "underweight_dia"
df.loc[(df["BMI"] < 18.5) & (df["Diabetes"] == 0), "NEW_BMI_DIABET"] = "underweight_no_dia"

df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Diabetes"] == 1), "NEW_BMI_DIABET"] = "healthy_dia"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Diabetes"] == 0), "NEW_BMI_DIABET"] = "healthy_no_dia"

df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Diabetes"] == 1), "NEW_BMI_DIABET"] = "overweight_dia"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Diabetes"] == 0), "NEW_BMI_DIABET"] = "overweight_no_dia"

df.loc[(df["BMI"] >= 30) & (df["Diabetes"] == 1), "NEW_BMI_DIABET"] = "obese_dia"
df.loc[(df["BMI"] >= 30) & (df["Diabetes"] == 0), "NEW_BMI_DIABET"] = "obese_no_dia"

# Cholesterol classification
df["New_Cholesterol"] = pd.cut(x=df['Cholesterol'], bins=[0, 200, 239, 420], labels=["Desirable", "Borderline High", "High"])

df['New_Activity'] = df['Exercise Hours Per Week'] * df['Physical Activity Days Per Week']

# Triglycerides classification
df["New_Triglycerides"] = pd.cut(x=df['Triglycerides'], bins=[0, 150, 199, 499, 810], labels=["Normal", "Borderline High", "High", "Very High"])

df["Active_hours_per_day"] = (24 - (df["Sedentary Hours Per Day"] + df["Sleep Hours Per Day"]))


df["New_Triglycerides"] = df["New_Triglycerides"].astype(object)
df["New_Cholesterol"] = df["New_Cholesterol"].astype(object)
df["NEW_BMI"] = df["NEW_BMI"].astype(object)


# Blood Press Difference
df["Blood_Pres_Dif"] = df["Systolic_Blood_Pres"] - df["Diastolic_Blood_Pres"]

# Blood Pressure Classification
df.loc[(df["Systolic_Blood_Pres"] < 120) & (df["Diastolic_Blood_Pres"] < 80), "New_Blood_Pres"] = "optimal"
df.loc[(df["Systolic_Blood_Pres"] < 130) & (df["Diastolic_Blood_Pres"] < 85), "New_Blood_Pres"] = "normal"
df.loc[(df["Systolic_Blood_Pres"] >= 130) & (df["Systolic_Blood_Pres"] < 140) & (df["Diastolic_Blood_Pres"] >= 85) & (df["Diastolic_Blood_Pres"] < 90), "New_Blood_Pres"] = "high-normal"
df.loc[(df["Systolic_Blood_Pres"] >= 140) & (df["Systolic_Blood_Pres"] < 160) & (df["Diastolic_Blood_Pres"] >= 90) & (df["Diastolic_Blood_Pres"] < 100), "New_Blood_Pres"] = "hyper_stage1"
df.loc[(df["Systolic_Blood_Pres"] >= 160) & (df["Systolic_Blood_Pres"] < 180) & (df["Diastolic_Blood_Pres"] >= 100) & (df["Diastolic_Blood_Pres"] < 110), "New_Blood_Pres"] = "hyper_stage2"
df.loc[(df["Systolic_Blood_Pres"] >= 180) & (df["Diastolic_Blood_Pres"] >= 100), "New_Blood_Pres"] = "hyper_stage3"

# Heart Rate Classification
df.loc[(df["Heart Rate"] >= 40) & (df["Heart Rate"] < 60), "New_Heart_Rate"] = "low"
df.loc[(df["Heart Rate"] >= 60) & (df["Heart Rate"] < 100), "New_Heart_Rate"] = "normal"
df.loc[(df["Heart Rate"] >= 100), "New_Heart_Rate"] = "high"

# Sleep Classification
df.loc[(df["Sleep Hours Per Day"] < 9) & (df["Sleep Hours Per Day"] >= 6), "New_Sleep_Score"] = "high_quality"
df.loc[(df["Sleep Hours Per Day"] < 6), "New_Sleep_Score"] = "low_quality"
df.loc[(df["Sleep Hours Per Day"] >= 9), "New_Sleep_Score"] = "very_low_quality"

## Mapping for sleep
sleep_mapping = {"very_low_quality": 1, "low_quality": 2, "high_quality": 3}
df["New_Sleep_Score"] = df["New_Sleep_Score"].map(sleep_mapping)

df["New_Strees*Sleep"] = df["New_Sleep_Score"] * df["Stress Level"]


cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Label Encoding #
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)


# One-Hot Encoding

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe

pre_ohe_cols = [col for col in cat_cols if 22 >= df[col].nunique() > 2]
ohe_cols = [col for col in pre_ohe_cols if ("Diet" not in col) and ("New_Sleep_Score" not in col)]

df = one_hot_encoder(df, ohe_cols)

##################### DENEMEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE

y = df["Heart Attack Risk"]
X = df.drop(["Heart Attack Risk"], axis=1)

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state = 50)
X_resample, y_resample = smote.fit_resample(X, y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resample)

smote = SMOTE(random_state = 50)
X_resample, y_resample = smote.fit_resample(X, y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resample)

X_train, X_test, y_train, y_test = train_test_split \
    (X_scaled, y_resample, test_size=0.33, random_state=42)

import numpy as np
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)


# Modelleri eğitelim ve test edelim

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Classifier:")
print(classification_report(y_test, y_pred_rf))

def plot_importance(model, features, num=len(X), save=False):
    # Features bir NumPy dizisi ise, sütun isimlerini X'ten alalım
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame(features, columns=X.columns)

    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train, num=20)

# Support Vector Classifier (SVC)
svc_model = SVC(random_state=42)
svc_model.fit(X_train, y_train)
y_pred_svc = svc_model.predict(X_test)
print("Support Vector Classifier:")
print(classification_report(y_test, y_pred_svc))


# Gradient Boosting Machine (GBM)
gbm_model = GradientBoostingClassifier(random_state=42)
gbm_model.fit(X_train, y_train)
y_pred_gbm = gbm_model.predict(X_test)
print("Gradient Boosting Machine (GBM):")
print(classification_report(y_test, y_pred_gbm))

plot_importance(gbm_model, X_train, num=20)

# XGBoost
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost:")
print(classification_report(y_test, y_pred_xgb))

plot_importance(xgb_model, X_train, num=20)

# LightGBM
lgbm_model = LGBMClassifier(random_state=42)
lgbm_model.fit(X_train, y_train)
y_pred_lgbm = lgbm_model.predict(X_test)
print("LightGBM:")
print(classification_report(y_test, y_pred_lgbm))

plot_importance(lgbm_model, X_train, num=20)

# K-Nearest Neighbors (KNN) Classifier
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print("K-Nearest Neighbors (KNN) Classifier:")
print(classification_report(y_test, y_pred_knn))


# Classification and Regression Trees (CART) Classifier
cart_model = DecisionTreeClassifier(random_state=42)
cart_model.fit(X_train, y_train)
y_pred_cart = cart_model.predict(X_test)
print("Classification and Regression Trees (CART) Classifier:")
print(classification_report(y_test, y_pred_cart))

plot_importance(cart_model, X_train, num=20)

######################## Hyperparametre Optimization ######################################

from sklearn.model_selection import GridSearchCV

# Random Forest için hiperparametre arama uzayı
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# SVC için hiperparametre arama uzayı
svc_param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf', 'poly']
}

# GBM için hiperparametre arama uzayı
gbm_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# XGBoost için hiperparametre arama uzayı
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# LightGBM için hiperparametre arama uzayı
lgbm_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'num_leaves': [31, 50, 100],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# KNN için hiperparametre arama uzayı
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]
}

# CART için hiperparametre arama uzayı
cart_param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

#knn_params = {"n_neighbors": range(2, 50)}

#cart_params = {'max_depth': range(1, 20),
               #"min_samples_split": range(2, 30)}

# Grid Search CV ile hiperparametre optimizasyonu
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5)
rf_grid_search.fit(X_train, y_train)


gbm_grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), gbm_param_grid, cv=5)
gbm_grid_search.fit(X_train, y_train)

xgb_grid_search = GridSearchCV(XGBClassifier(random_state=42), xgb_param_grid, cv=5)
xgb_grid_search.fit(X_train, y_train)

lgbm_grid_search = GridSearchCV(LGBMClassifier(random_state=42), lgbm_param_grid, cv=5)
lgbm_grid_search.fit(X_train, y_train)

knn_grid_search = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5)
knn_grid_search.fit(X_train, y_train)

cart_grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), cart_param_grid, cv=5)
cart_grid_search.fit(X_train, y_train)

# En iyi parametreler
print("Random Forest En İyi Parametreler:", rf_grid_search.best_params_)
print("GBM En İyi Parametreler:", gbm_grid_search.best_params_)
print("XGBoost En İyi Parametreler:", xgb_grid_search.best_params_)
print("LightGBM En İyi Parametreler:", lgbm_grid_search.best_params_)
print("K-Nearest Neighbors (KNN) En İyi Parametreler:", knn_grid_search.best_params_)
print("Classification and Regression Trees (CART) En İyi Parametreler:", cart_grid_search.best_params_)

# En iyi modelleri alın
best_rf_model = rf_grid_search.best_estimator_
best_gbm_model = gbm_grid_search.best_estimator_
best_xgb_model = xgb_grid_search.best_estimator_
best_lgbm_model = lgbm_grid_search.best_estimator_
best_knn_model = knn_grid_search.best_estimator_
best_cart_model = cart_grid_search.best_estimator_

# Test verisi üzerinde en iyi modelleri değerlendirin
y_pred_rf = best_rf_model.predict(X_test)
print("Random Forest:")
print(classification_report(y_test, y_pred_rf))

y_pred_gbm = best_gbm_model.predict(X_test)
print("GBM:")
print(classification_report(y_test, y_pred_gbm))

y_pred_xgb = best_xgb_model.predict(X_test)
print("XGBoost:")
print(classification_report(y_test, y_pred_xgb))

y_pred_lgbm = best_lgbm_model.predict(X_test)
print("LightGBM:")
print(classification_report(y_test, y_pred_lgbm))

y_pred_knn = knn_grid_search.best_estimator_.predict(X_test)
print("K-Nearest Neighbors (KNN):")
print(classification_report(y_test, y_pred_knn))

y_pred_cart = cart_grid_search.best_estimator_.predict(X_test)
print("Classification and Regression Trees (CART):")
print(classification_report(y_test, y_pred_cart))

