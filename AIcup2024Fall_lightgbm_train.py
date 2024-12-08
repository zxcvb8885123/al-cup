import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
import lightgbm as lgb
import joblib

#讀取數據
data = pd.read_csv(r'D:\program\python\ML\\AIcup2024FalltrainData.csv')

#特徵與目標
X = data[['year', 'month', 'day', 'hour', 'minute', 'weekday', 'LocationCode','hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'month_sin', 'month_cos','hour_squared','quarter','time_of_day','day_of_year','week_of_year']]
y = data['Power(mW)']

#將數據分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#指定類別型特徵
cat_features = ['year', 'month', 'day', 'hour', 'minute', 'weekday','LocationCode','quarter','time_of_day']

#設置 LightGBM 模型參數
params = {
    'objective': 'regression',   # 回歸
    'metric': 'rmse',            # 評估指標
    'learning_rate': 0.01,       # 學習率
    'max_depth': 25,             # 最大樹深度
    'num_leaves': 355,           # 樹葉節點數
    'feature_fraction': 0.8,     # 每次使用部分特徵
    'bagging_fraction': 0.8,     # 每次使用部分數據
    'bagging_freq': 5,           # 每5次疊代重新抽樣數據
    'verbose': -1                # 禁止冗長輸出
}

#訓練第一個模型
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
test_data = lgb.Dataset(X_test, label=y_test, categorical_feature=cat_features, reference=train_data)

#num_boost_round=1500
lgb_reg1 = lgb.train(
    params=params,
    train_set=train_data,             
    valid_sets=[test_data],           
    num_boost_round=1500,             
    callbacks=[lgb.log_evaluation(period=50)]
)

#使用第一個模型對測試集進行預測
y_pred1 = lgb_reg1.predict(X_test, num_iteration=lgb_reg1.best_iteration)

#使用第一個模型的預測結果作為第二個模型的額外特徵
X_train_meta = X_train.copy()
X_test_meta = X_test.copy()
X_train_meta['pred1'] = lgb_reg1.predict(X_train)
X_test_meta['pred1'] = y_pred1

#訓練第二個模型，使用第一個模型的預測作為特徵之一
train_data_meta = lgb.Dataset(X_train_meta, label=y_train, categorical_feature=cat_features)
test_data_meta = lgb.Dataset(X_test_meta, label=y_test, categorical_feature=cat_features, reference=train_data_meta)

#num_boost_round=700
lgb_reg2 = lgb.train(
    params=params,
    train_set=train_data_meta,             
    valid_sets=[test_data_meta],           
    num_boost_round=700,             
    callbacks=[lgb.log_evaluation(period=50)]
)

#預測
y_pred2 = lgb_reg2.predict(X_test_meta, num_iteration=lgb_reg2.best_iteration)

#評估指標
mse = mean_squared_error(y_test, y_pred2)
mae = mean_absolute_error(y_test, y_pred2)
r2 = r2_score(y_test, y_pred2)
rmse = sqrt(mse)

#結果
print(f'均方誤差 (MSE): {mse}')
print(f'平均絕對值誤差 (MAE): {mae}')
print(f'R² 分數: {r2}')
print(f'均方根誤差 (RMSE): {rmse}')

#保存模型
joblib.dump(lgb_reg1, r'D:\program\python\ML\AIcup2024Fall_lightgbm_1.pkl')
joblib.dump(lgb_reg2, r'D:\program\python\ML\AIcup2024Fall_lightgbm_2.pkl')

print('模型已保存')