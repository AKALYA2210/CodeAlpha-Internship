import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("Unemployment_Rate.csv")

print(df.head())
print(df.info())
df['Date'] = pd.to_datetime(df['Date'])
print(df.isnull().sum())
df = df.dropna()
print(df.describe())
plt.figure()
plt.plot(df['Date'], df['Unemployment Rate'], marker='o')
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.title("Unemployment Trend Over Time")
plt.xticks(rotation=45)
plt.show()
covid_data = df[(df['Date'] >= "2020-01-01") & (df['Date'] <= "2021-12-31")]

plt.figure()
plt.plot(covid_data['Date'], covid_data['Unemployment Rate'], marker='o')
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.title("Unemployment During COVID-19")
plt.xticks(rotation=45)
plt.show()
df['Month'] = df['Date'].dt.month

monthly_avg = df.groupby('Month')['Unemployment Rate'].mean()

plt.figure()
monthly_avg.plot(kind='bar')
plt.xlabel("Month")
plt.ylabel("Average Unemployment Rate")
plt.title("Monthly Unemployment Trend")
plt.show()
