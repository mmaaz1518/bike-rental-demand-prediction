import numpy as np
import pandas as pd
df=pd.read_csv('hour.csv')
print(df.head())
print(df.columns)
import pandas as pd

df = pd.read_csv("hour.csv")

# 2. Check missing values
print("Missing Values:\n", df.isnull().sum())

df = df.drop_duplicates()

# 4. Convert date column to datetime
df["dteday"] = pd.to_datetime(df["dteday"])

# 5. Remove unnecessary columns
df = df.drop(columns=[
    "instant",
    "dteday",
    "casual",
    "registered"
])
# 6. Handle missing values, if any
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

# 7. Encode categorical features
categorical_columns = [
    "season",
    "mnth",
    "hr",
    "weekday",
    "weathersit"
]
df = pd.get_dummies(
    df,
    columns=categorical_columns,
    dtype=int
)
# 8. Separate features and target
X = df.drop(columns=["cnt"])
y = df["cnt"]
print("\nFinal cleaned dataset:")
print(df.head())

print("\nFinal dataset shape:", df.shape)
print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\nPhase 3 completed successfully!")
# print(df.columns.tolist())


import matplotlib.pyplot as plt
import seaborn as sns

df.columns = df.columns.str.strip()

sns.set_theme(style="whitegrid")

print("=" * 50)
print("BIKE RENTAL DATASET - EDA")
print("=" * 50)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())



print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nTotal missing values:", df.isnull().sum().sum())
print("Total duplicate rows:", df.duplicated().sum())


plt.figure(figsize=(10, 5))

if "hr" in df.columns:
    hourly_data = df.groupby("hr")["cnt"].mean()
    plt.plot(hourly_data.index, hourly_data.values, marker="o")
    plt.xticks(range(24))
else:
    # Works when the hour column has been one-hot encoded
    hour_columns = [
        f"hr_{hour}" for hour in range(24)
        if f"hr_{hour}" in df.columns
    ]

    if hour_columns:
        hourly_data = {
            int(col.split("_")[1]): df.loc[df[col] == 1, "cnt"].mean()
            for col in hour_columns
        }
        hourly_data = pd.Series(hourly_data).sort_index()
        plt.plot(hourly_data.index, hourly_data.values, marker="o")
        plt.xticks(range(24))
    else:
        print("Hour information is unavailable; skipping hourly plot.")
        hourly_data = None

if "hr" in df.columns or any(f"hr_{h}" in df.columns for h in range(24)):
    plt.title("Average Bike Rentals by Hour")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Average Rentals")
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(8, 5))

if "season" in df.columns:
    season_data = df.groupby("season")["cnt"].mean()
    plt.bar(season_data.index.astype(str), season_data.values)
    plt.xlabel("Season")
    plt.ylabel("Average Rentals")
    plt.title("Average Bike Rentals by Season")
    plt.tight_layout()
    plt.show()

else:
    season_columns = [
        f"season_{i}" for i in range(1, 5)
        if f"season_{i}" in df.columns
    ]

    if season_columns:
        season_data = {
            col.replace("season_", "Season "):
            df.loc[df[col] == 1, "cnt"].mean()
            for col in season_columns
        }
        plt.bar(season_data.keys(), season_data.values())
        plt.xlabel("Season")
        plt.ylabel("Average Rentals")
        plt.title("Average Bike Rentals by Season")
        plt.tight_layout()
        plt.show()
    else:
        print("Season information is unavailable; skipping season plot.")

plt.figure(figsize=(8, 5))

if "weathersit" in df.columns:
    weather_data = df.groupby("weathersit")["cnt"].mean()
    plt.bar(weather_data.index.astype(str), weather_data.values)
    plt.xlabel("Weather Situation")
    plt.ylabel("Average Rentals")
    plt.title("Average Bike Rentals by Weather")
    plt.tight_layout()
    plt.show()

else:
    weather_columns = [
        f"weathersit_{i}" for i in range(1, 5)
        if f"weathersit_{i}" in df.columns
    ]

    if weather_columns:
        weather_data = {
            col.replace("weathersit_", "Weather "):
            df.loc[df[col] == 1, "cnt"].mean()
            for col in weather_columns
        }
        plt.bar(weather_data.keys(), weather_data.values())
        plt.xlabel("Weather Situation")
        plt.ylabel("Average Rentals")
        plt.title("Average Bike Rentals by Weather")
        plt.tight_layout()
        plt.show()
    else:
        print("Weather information is unavailable; skipping weather plot.")


plt.figure(figsize=(9, 5))

if "weekday" in df.columns:
    weekday_data = df.groupby("weekday")["cnt"].mean()
    plt.bar(weekday_data.index.astype(str), weekday_data.values)
    plt.xlabel("Weekday (0 = Sunday)")
    plt.ylabel("Average Rentals")
    plt.title("Average Bike Rentals by Weekday")
    plt.tight_layout()
    plt.show()

else:
    weekday_columns = [
        f"weekday_{i}" for i in range(7)
        if f"weekday_{i}" in df.columns
    ]

    if weekday_columns:
        weekday_data = {
            col.replace("weekday_", "Day "):
            df.loc[df[col] == 1, "cnt"].mean()
            for col in weekday_columns
        }
        plt.bar(weekday_data.keys(), weekday_data.values())
        plt.xlabel("Weekday (0 = Sunday)")
        plt.ylabel("Average Rentals")
        plt.title("Average Bike Rentals by Weekday")
        plt.tight_layout()
        plt.show()
    else:
        print("Weekday information is unavailable; skipping weekday plot.")



if "temp" in df.columns:
    plt.figure(figsize=(9, 5))

    sns.scatterplot(data=df, x="temp", y="cnt", alpha=0.25)

    plt.title("Temperature vs Bike Rentals")
    plt.xlabel("Normalized Temperature")
    plt.ylabel("Total Bike Rentals")
    plt.tight_layout()
    plt.show()
else:
    print("Temperature column is unavailable; skipping temperature plot.")


if "hum" in df.columns:
    plt.figure(figsize=(9, 5))

    sns.scatterplot(data=df, x="hum", y="cnt", alpha=0.25)

    plt.title("Humidity vs Bike Rentals")
    plt.xlabel("Normalized Humidity")
    plt.ylabel("Total Bike Rentals")
    plt.tight_layout()
    plt.show()
else:
    print("Humidity column is unavailable; skipping humidity plot.")


numeric_df = df.select_dtypes(include="number")
correlation = numeric_df.corr()

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0,
    annot=False
)

plt.title("Correlation Heatmap of Bike Rental Features")
plt.tight_layout()
plt.show()

if "cnt" in correlation.columns:
    target_correlation = correlation["cnt"].drop("cnt").sort_values()

    print("\nFeature correlations with bike rentals (cnt):")
    print(target_correlation.sort_values(ascending=False))

    plt.figure(figsize=(10, 7))

    target_correlation.plot(kind="barh")

    plt.title("Feature Correlation with Bike Rentals")
    plt.xlabel("Correlation with cnt")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

print("\n" + "=" * 50)
print("EDA SUMMARY")
print("=" * 50)

print("Total records:", len(df))
print("Total columns:", len(df.columns))

if "cnt" in df.columns:
    print("Average hourly rentals:", round(df["cnt"].mean(), 2))
    print("Minimum hourly rentals:", df["cnt"].min())
    print("Maximum hourly rentals:", df["cnt"].max())

print("\nEDA completed successfully!")


# PHASE 5: LINEAR REGRESSION MODEL

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X = df.drop(columns=["cnt"])
y = df["cnt"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

model = LinearRegression()
# 5. Train the model
model.fit(X_train, y_train)

# 6. Make predictions on the test data
y_pred = model.predict(X_test)

results = pd.DataFrame({
    "Actual Rentals": y_test.values,
    "Predicted Rentals": y_pred.round(2)
})
print("\nActual vs Predicted Bike Rentals:")
print(results.head(10))

print("\nModel Intercept:", round(model.intercept_, 2))
print("Number of model coefficients:", len(model.coef_))



# PHASE 6: MODEL EVALUATION

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
y_pred = model.predict(X_test)
# 2. Calculate regression evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("=" * 45)
print("BIKE RENTAL MODEL EVALUATION")
print("=" * 45)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")
print(f"R-squared percentage: {r2 * 100:.2f}%")

# 4. Create Actual vs Predicted DataFrame
results = pd.DataFrame({
    "Actual Rentals": y_test.values,
    "Predicted Rentals": y_pred
})
print("\nFirst 10 Actual vs Predicted Values:")
print(results.head(10).round(2))

# 5. Chart: Actual vs Predicted Rentals
plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5,
    label="Predictions"
)

# Ideal prediction line
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    label="Perfect Predictions"
)
plt.xlabel("Actual Bike Rentals")
plt.ylabel("Predicted Bike Rentals")
plt.title("Actual vs Predicted Bike Rental Demand")
plt.legend()
plt.tight_layout()
plt.show()
# 6. Chart: Compare actual and predicted values for 30 test samples
comparison = results.head(30).reset_index(drop=True)
plt.figure(figsize=(12, 6))
plt.plot(
    comparison.index,
    comparison["Actual Rentals"],
    marker="o",
    label="Actual Rentals"
)
plt.plot(
    comparison.index,
    comparison["Predicted Rentals"],
    marker="x",
    label="Predicted Rentals"
)

plt.xlabel("Test Sample")
plt.ylabel("Number of Bike Rentals")
plt.title("Actual vs Predicted Rentals (First 30 Test Samples)")
plt.legend()
plt.tight_layout()
plt.show()


