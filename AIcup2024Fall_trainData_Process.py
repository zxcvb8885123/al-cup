import pandas as pd
import numpy as np

#讀取數據
data1 = pd.read_csv(r'D:\program\python\ML\L2_Train_2.csv')
data2 = pd.read_csv(r'D:\program\python\ML\L4_Train_2.csv')
data3 = pd.read_csv(r'D:\program\python\ML\L7_Train_2.csv')
data4 = pd.read_csv(r'D:\program\python\ML\L8_Train_2.csv')
data5 = pd.read_csv(r'D:\program\python\ML\L9_Train_2.csv')
data6 = pd.read_csv(r'D:\program\python\ML\L10_Train_2.csv')
data7 = pd.read_csv(r'D:\program\python\ML\L12_Train_2.csv')
data8 = pd.concat([pd.read_csv(f'D:\\program\\python\\ML\\L{i}_Train.csv') for i in range(1, 18)]).dropna()

data = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8]).dropna()
print(f'原來: {data.shape}')

#篩選數據
filterData = data[(data['Pressure(hpa)'] <= 1200) & (data['Humidity(%)'] <= 100) & (data['Sunlight(Lux)'] < 117758.2)]

#轉換日期時間格式
filterData['DateTime'] = pd.to_datetime(filterData['DateTime'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')

#提取時間特徵
filterData['year'] = filterData['DateTime'].dt.year
filterData['month'] = filterData['DateTime'].dt.month
filterData['day'] = filterData['DateTime'].dt.day
filterData['hour'] = filterData['DateTime'].dt.hour
filterData['minute'] = filterData['DateTime'].dt.minute
filterData['weekday'] = filterData['DateTime'].dt.weekday  # 0=Monday, 6=Sunday

#時間周期性特徵
filterData['hour_sin'] = np.sin(2 * np.pi * filterData['hour'] / 24)
filterData['hour_cos'] = np.cos(2 * np.pi * filterData['hour'] / 24)
filterData['minute_sin'] = np.sin(2 * np.pi * filterData['minute'] / 60)
filterData['minute_cos'] = np.cos(2 * np.pi * filterData['minute'] / 60)
filterData['month_sin'] = np.sin(2 * np.pi * filterData['month'] / 12)
filterData['month_cos'] = np.cos(2 * np.pi * filterData['month'] / 12)

#季節（0春季，1夏季，2秋季，3冬季）
filterData['quarter'] = filterData['month'] % 12 // 3

#每年中的第幾天
filterData['day_of_year'] = filterData['DateTime'].dt.dayofyear

#周數
filterData['week_of_year'] = filterData['DateTime'].dt.isocalendar().week

#小時數的平方
filterData['hour_squared'] = filterData['hour'] ** 2

#白天/下午/晚上
filterData['time_of_day'] = filterData['hour'].apply(lambda x: 0 if 6 <= x < 12 else (1 if 12 <= x < 18 else 2))  #白天/下午/晚上

#儲存處理後的數據
filterData.to_csv(r'D:\program\python\ML\AIcup2024FalltrainData.csv', index=False)
print(f'處理後: {filterData.shape}')
print('save')