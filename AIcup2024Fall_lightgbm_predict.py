import joblib
import pandas as pd
import numpy as np

#載入訓練好的兩個模型
lgbm_model1 = joblib.load(r'D:\program\python\ML\AIcup2024Fall_lightgbm_1.pkl')
lgbm_model2 = joblib.load(r'D:\program\python\ML\AIcup2024Fall_lightgbm_2.pkl')

#讀取資料
data = pd.read_csv(r'D:\program\python\ML\AIcup2024FalluploadData.csv')
df_pred = pd.read_csv(r'D:\program\python\ML\upload(no answer).csv')

#特徵
X = data[['year', 'month', 'day', 'hour', 'minute','weekday', 'LocationCode','hour_sin','hour_cos','minute_sin','minute_cos','month_sin','month_cos','hour_squared','quarter','time_of_day','day_of_year','week_of_year']]

#使用第一個模型進行預測
y_pred1 = lgbm_model1.predict(X)

#限制第一個模型的預測結果為非負數
y_pred1 = np.clip(y_pred1, 0, None)

#把第一個模型的預測結果作為第二個模型的額外特徵
X_meta = X.copy()
X_meta['pred1'] = y_pred1

#使用第二個模型進行最終預測
y_pred2 = lgbm_model2.predict(X_meta)

#限制第二個模型的預測結果為非負數
y_pred2 = np.clip(y_pred2, 0, None)

#將最終預測結果加入 '答案' 欄位
df_pred['答案'] = y_pred2

#保存結果回原來的 CSV 文件
df_pred.to_csv(r'D:\program\python\ML\upload(no answer).csv', index=False, header=True)

print('預測結果已成功儲存回upload.csv')