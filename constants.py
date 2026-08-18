# constants.py

# 我們的預測目標
TRAFFIC_TARGETS = ["Jam_Factor", "Speed_kmh"]

# 輸入給模型的特徵 (包含交通、地理、時間與天氣)
# 注意：依照你的截圖，我們選用已有的特徵
BASE_FEATURES = [
    "Jam_Factor",
    "Speed_kmh",
    "Length_m",       # 使用標準化後的長度
    "Latitude",
    "Longitude",
    "Holiday",
    "Temperature",
    "Humidity",
    "Wind_Speed",
    "UV_Index",
    "Pressure",
    "Visibility",
    "Dew_Point",
    "time_sin",
    "time_cos",
    "day_sin",
    "day_cos",
    # 這裡預留天氣的 One-hot 編碼欄位，根據你的 CSV 實際有的欄位填寫
    "Description_Cloudy",
    # "Description_Clear", # 視你的 CSV 而定
]