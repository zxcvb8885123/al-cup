# AI CUP 2024 FAII PREDICTS POWER


## 運行環境

<h3>
    <img src="https://img.shields.io/badge/Python3.11.0-FFD43B?style=for-the-badge&logo=python&logoColor=blue" alt="pythpn Badge">
    <img src="https://img.shields.io/badge/Pandas2.2.3-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" alt="pandas Badge">
    <img src="https://img.shields.io/badge/scikit_learn1.5.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn Badge">
    <img src="https://img.shields.io/badge/Lightgbm4.5.0-792DE4?style=for-the-badge&logoColor=white" alt="Lightgbm Badge">
    <img src="https://img.shields.io/badge/joblib1.4.2-black?style=for-the-badge" alt="joblib1">
    <img src="https://img.shields.io/badge/Numpy1.26.4-777BB4?style=for-the-badge&logo=numpy&logoColor=white" alt="numpy">
    <img src="https://img.shields.io/badge/Visual%20Studio%20Code1.95.3-%237df9ff?style=for-the-badge" alt="Visual Studio Code">
    <img src="https://img.shields.io/badge/Notepad++8.7.1-90E59A?style=for-the-badge&logo=notepadplusplus&logoColor=white" alt="Notepad++ Badge">
<h3>

## 資料
- `AIcup2024FalluploadData.csv`: upload的資料
- `AIcup2024FalltrainData.csv`: 訓練的資料
- `AIcup2024FalltestData.csv`: 測試的資料
- `upload(no answer).csv`: 最後預測的結果

處理腳本
- `AIcup2024Fall_trainData_Process.py`:L1~L17資料合併，並增加時間週期性特徵
- `AIcup2024Fall_uploadData_Process.py`: 處理upload的資料，並增加時間週期性特徵
- `AIcup2024Fall_testData_process.py`:處理test的資料，並增加時間週期性特徵

## 訓練
- `AIcup2024Fall_lightgbm_train.py`:
  - `訓練兩層 LightGBM 模型`
    - `第一層模型用於初步預測。`
    - `第二層模型將第一層的預測結果作為額外特徵，進行進一步預測。`
  - `支援類別型特徵，提升對類別數據的處理能力`
  - `模型表現:均方誤差 (MSE)、平均絕對值誤差 (MAE)、R² 分數、均方根誤差 (RMSE)`
  - `保存訓練完成的模型以供後續使用`

設置 LightGBM 模型參數
- `params` = {
    - `'objective'`: 'regression',   # 回歸
    - `'metric'`: 'rmse',            # 評估指標
    - `'learning_rate'`: 0.01,       # 學習率
    - `'max_depth'`: 25,             # 最大樹深度
    - `'num_leaves'`: 355,           # 樹葉節點數
    - `'feature_fraction'`: 0.8,     # 每次使用部分特徵
    - `'bagging_fraction'`: 0.8,     # 每次使用部分數據
    - `'bagging_freq'`: 5,           # 每5次疊代重新抽樣數據
    - `'verbose'`: -1                # 禁止冗長輸出
}

使用AIcup2024FalltrainData.csv進行訓練，並保存模型

## 預測
- `AIcup2024Fall_lightgbm_predict.py`:
    - `載入模型`
      - `加載已訓練完成的第一層和第二層 LightGBM 模型`
    - `讀取數據`
      - `從 AIcup2024FalluploadData.csv 載入需要預測的特徵數據`
      - `從 upload(no answer).csv 載入結果模板`
    - `進行兩層預測`
      - `第一層模型預測，並對結果進行非負限制`
      - `第一層模型的預測作為第二層模型的額外特徵，進行最終預測`

最後保存預測檔案

## 程式區塊介紹

### 測試程式
| class | Description | Output File |
|:-----:| :---------: | :---------: |
| AIcup2024Fall_lightgbm_predict_test.py | 預測大致誤差 | 生成一個新的csv |

### 資料處理程式:data
| class | Description | Output File |
|:-----:| :---------: | :---------: |
| AIcup2024Fall_trainData_Process.py | 前處理訓練資料 | AIcup2024FalltrainData.csv |
| AIcup2024Fall_uploadData_Process.py | 前處理upload的資料 | AIcup2024FalluploadData.csv |
| AIcup2024Fall_testData_process.py | 前處理test的資料 | AIcup2024FalltestData.csv |

### 訓練程式/預測程式:model
| class | Description | Output File |
|:-----:| :---------: | :---------: |
| AIcup2024Fall_lightgbm_predict.py | 預測power |  upload(no answer).csv|
| AIcup2024Fall_lightgbm_train.py | 訓練modle | AIcup2024Fall_lightgbm_1.pkl,AIcup2024Fall_lightgbm_2.pkl |
