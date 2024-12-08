import joblib
import pandas as pd
import numpy as np

#讀取已訓練的兩個模型
model1 = joblib.load(r'D:\program\python\ML\AIcup2024Fall_lightgbm_1.pkl')
model2 = joblib.load(r'D:\program\python\ML\AIcup2024Fall_lightgbm_2.pkl')

#讀取新的數據進行預測
data = pd.read_csv(r'D:\program\python\ML\AIcup2024FalltestData.csv')

#特徵
X = data[['year', 'month', 'day', 'hour', 'minute','weekday', 'LocationCode','hour_sin','hour_cos','minute_sin','minute_cos','month_sin','month_cos','hour_squared','quarter','time_of_day','day_of_year','week_of_year']]

#使用第一個模型進行預測
y_pred1 = model1.predict(X)

#把第一個模型的預測結果當作第二個模型的額外特徵
X_meta = X.copy()
X_meta['pred1'] = y_pred1

#使用第二個模型進行最終預測
y_pred2 = model2.predict(X_meta)

#保證預測結果不為負數
y_pred2 = np.clip(y_pred2, 0, None)

#打印預測結果
print(y_pred2)

#建立預測結果 DataFrame
df_pred = data[['Power(mW)']].copy() 
df_pred['Predicted_Power'] = y_pred2
df_pred['error'] = abs(df_pred['Power(mW)'] - df_pred['Predicted_Power'])

#計算平均誤差
print('平均誤差:', df_pred['error'].mean())

#保存預測結果到 Excel 文件
df_pred.to_excel(r'D:\program\python\ML\AIcup2024PredictTest.xlsx', index=False)