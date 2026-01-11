import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')

target = df["Diabetes_binary"]

offline_df, online_df = train_test_split(df, test_size=0.2, random_state=42, stratify=target)
offline_df.to_csv('offline.csv', index=False)
online_df.to_csv('online.csv', index=False)

