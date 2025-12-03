import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Task 1: Data Acquisition & Loading ----------
# Change the path & filename as needed
data_path = "data/raw_weather.csv"

df = pd.read_csv(data_path)

print("Head:")
print(df.head())
print("\nInfo:")
print(df.info())
print("\nDescribe:")
print(df.describe())

# ---------- Task 2: Data Cleaning & Processing ----------
# 1) Convert date column
# Change 'Date' to the actual column name in your CSV
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# 2) Handle missing values (simple example: drop rows with NaN in key columns)
key_cols = ['Date', 'Temperature', 'Rainfall', 'Humidity']  # adjust names
df = df.dropna(subset=key_cols)

# 3) Filter relevant columns
df = df[['Date', 'Temperature', 'Rainfall', 'Humidity']]

# 4) Set Date as index (useful for resampling)
df = df.set_index('Date').sort_index()

# Save cleaned data
df.to_csv("data/cleaned_weather.csv")

# ---------- Task 3: Statistical Analysis with NumPy ----------
# Daily stats (already daily if each row is a day)
daily_stats = df.agg(['mean', 'min', 'max', 'std'])

print("\nDaily stats:")
print(daily_stats)

# Monthly stats
monthly = df.resample('M').agg(['mean', 'min', 'max', 'std'])
print("\nMonthly stats (head):")
print(monthly.head())

# Yearly stats
yearly = df.resample('Y').agg(['mean', 'min', 'max', 'std'])
print("\nYearly stats (head):")
print(yearly.head())

# You can also use NumPy directly
temps = df['Temperature'].values
print("\nNumPy stats for Temperature:")
print("Mean:", np.mean(temps))
print("Min:", np.min(temps))
print("Max:", np.max(temps))
print("Std:", np.std(temps))

# ---------- Task 4: Visualization with Matplotlib ----------

# 1) Line chart for daily temperature trends
plt.figure()
df['Temperature'].plot()
plt.title("Daily Temperature Trend")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.tight_layout()
plt.savefig("plots/daily_temperature_trend.png")
plt.close()

# 2) Bar chart for monthly rainfall totals
monthly_rain = df['Rainfall'].resample('M').sum()
plt.figure()
monthly_rain.plot(kind='bar')
plt.title("Monthly Rainfall Totals")
plt.xlabel("Month")
plt.ylabel("Rainfall")
plt.tight_layout()
plt.savefig("plots/monthly_rainfall_bar.png")
plt.close()

# 3) Scatter plot for humidity vs temperature
plt.figure()
plt.scatter(df['Temperature'], df['Humidity'])
plt.title("Humidity vs. Temperature")
plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.tight_layout()
plt.savefig("plots/humidity_vs_temperature_scatter.png")
plt.close()

# 4) Combined figure with subplots (Bonus)
fig, axes = plt.subplots(2, 1, figsize=(8, 8))

# Top: temperature line
axes[0].plot(df.index, df['Temperature'])
axes[0].set_title("Daily Temperature Trend")
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Temperature")

# Bottom: monthly rainfall bar
axes[1].bar(monthly_rain.index.strftime('%Y-%m'), monthly_rain.values)
axes[1].set_title("Monthly Rainfall Totals")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Rainfall")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("plots/combined_plots.png")
plt.close()

# ---------- Task 5: Grouping & Aggregation ----------
# Example: group by month number
df['month'] = df.index.month
df['year'] = df.index.year

monthly_grouped = df.groupby('month').agg({
    'Temperature': ['mean', 'min', 'max'],
    'Rainfall': 'sum',
    'Humidity': 'mean'
})
print("\nGrouped by month:")
print(monthly_grouped)

# Example: simple season grouping (you can tweak this)
def season_from_month(m):
    if m in [12, 1, 2]:
        return 'Winter'
    elif m in [3, 4, 5]:
        return 'Spring'
    elif m in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

df['season'] = df['month'].apply(season_from_month)

season_stats = df.groupby('season').agg({
    'Temperature': 'mean',
    'Rainfall': 'sum',
    'Humidity': 'mean'
})
print("\nSeasonal stats:")
print(season_stats)