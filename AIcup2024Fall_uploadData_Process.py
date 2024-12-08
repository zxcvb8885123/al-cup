import pandas as pd
import numpy as np

#讀取數據
data = pd.read_csv(r'D:\program\python\ML\upload(no answer).csv')

#把序號內的日期跟地點拆開
data['datetime_part'] = data['序號'].astype(str).str[:-1] 
data['LocationCode'] = data['序號'].astype(str).str[-2:]

#轉換日期時間格式
data['datetime_part'] = pd.to_datetime(data['datetime_part'], format='%Y%m%d%H%M%S', errors='coerce')

if data['datetime_part'].isna().sum() > 0:
    print("loss")

#提取時間特徵
data['year'] = data['datetime_part'].dt.year
data['month'] = data['datetime_part'].dt.month
data['day'] = data['datetime_part'].dt.day
data['hour'] = data['datetime_part'].dt.hour
data['minute'] = data['datetime_part'].dt.minute
data['weekday'] = data['datetime_part'].dt.weekday

#時間周期性特徵
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
data['minute_sin'] = np.sin(2 * np.pi * data['minute'] / 60)
data['minute_cos'] = np.cos(2 * np.pi * data['minute'] / 60)
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

#季度（0春季，1夏季，2秋季，3冬季）
data['quarter'] = data['month'] % 12 // 3

#每年中的第幾天
data['day_of_year'] = data['datetime_part'].dt.dayofyear

#周數
data['week_of_year'] = data['datetime_part'].dt.isocalendar().week

#小時數的平方
data['hour_squared'] = data['hour'] ** 2

#其他時間特徵
data['time_of_day'] = data['hour'].apply(lambda x: 0 if 6 <= x < 12 else (1 if 12 <= x < 18 else 2))  #白天/下午/晚上

filterData = data[['year', 'month', 'day', 'hour', 'minute','weekday', 'LocationCode','hour_sin','hour_cos','minute_sin','minute_cos','month_sin','month_cos','quarter','day_of_year','week_of_year','hour_squared','time_of_day']]

filterData.to_csv(r'D:\program\python\ML\AIcup2024FalluploadData.csv', index=False)
print('save')