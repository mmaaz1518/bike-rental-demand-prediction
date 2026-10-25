
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
# 1. PAGE SETTINGS
st.set_page_config(
    page_title="Bike Rental Demand Prediction",
    page_icon="🚲",
    layout="centered"
)
st.title("🚲 Bike Rental Demand Prediction")
st.write("Enter the conditions to estimate hourly bike rental demand.")
# 2. LOAD DATA AND TRAIN MODEL
DATA_PATH = Path(__file__).parent / "hour.csv"


@st.cache_resource
def train_model():

    # Load dataset
    data = pd.read_csv(DATA_PATH)

    # Remove unnecessary columns if they exist
    columns_to_drop = [
        "instant",
        "dteday",
        "casual",
        "registered"
    ]

    data = data.drop(
        columns=[
            col for col in columns_to_drop
            if col in data.columns
        ]
    )

    # Check that target column exists
    if "cnt" not in data.columns:
        raise ValueError(
            "The dataset must contain a 'cnt' target column."
        )

    # Separate features and target
    X = data.drop(columns=["cnt"])
    y = data["cnt"]

    # Convert any remaining non-numeric columns
    X = pd.get_dummies(X, dtype=int)

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    return model, X.columns.tolist()


try:
    model, feature_columns = train_model()

except FileNotFoundError:
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info(
        "Place hour.csv inside the data folder "
        "next to your app.py file."
    )
    st.stop()

except Exception as error:
    st.error(f"Could not train model: {error}")
    st.stop()


# 3. USER INPUTS

st.subheader("Time Conditions")

col1, col2 = st.columns(2)

with col1:
    year = st.selectbox("Year", [2011, 2012], index=1)
    month = st.selectbox("Month", list(range(1, 13)), index=5)
    hour = st.slider("Hour of day", 0, 23, 17)
    weekday = st.selectbox(
        "Weekday (0 = Sunday)", list(range(7)), index=2
    )

with col2:
    season = st.selectbox(
        "Season",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Spring",
            2: "Summer",
            3: "Fall",
            4: "Winter"
        }[x],
        index=1
    )

    weather = st.selectbox(
        "Weather",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Clear / partly cloudy",
            2: "Mist / cloudy",
            3: "Light rain or snow",
            4: "Heavy rain or snow"
        }[x]
    )

holiday = st.selectbox(
    "Holiday?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

workingday = st.selectbox(
    "Working day?",
    [0, 1],
    index=1,
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# 4. WEATHER MEASUREMENTS

st.subheader("Weather Measurements")

st.caption(
    "Temperature, feels-like temperature, humidity, "
    "and wind speed use the dataset's normalized 0–1 scale."
)

col3, col4 = st.columns(2)

with col3:
    temp = st.slider("Temperature", 0.0, 1.0, 0.60, 0.01)
    atemp = st.slider(
        "Feels-like temperature", 0.0, 1.0, 0.58, 0.01
    )

with col4:
    hum = st.slider("Humidity", 0.0, 1.0, 0.50, 0.01)
    windspeed = st.slider("Wind speed", 0.0, 1.0, 0.20, 0.01)


# 5. PREPARE INPUT FOR PREDICTION

if st.button("Predict Bike Demand", type="primary"):

    # Start with zero values for every training feature
    input_data = pd.DataFrame(
        np.zeros((1, len(feature_columns))),
        columns=feature_columns
    )

    # Original numerical features
    ordinary_values = {
        "yr": year - 2011,
        "holiday": holiday,
        "workingday": workingday,
        "temp": temp,
        "atemp": atemp,
        "hum": hum,
        "windspeed": windspeed
    }

    for column, value in ordinary_values.items():
        if column in input_data.columns:
            input_data.loc[0, column] = value

    # Original categorical features or one-hot encoded features
    category_values = {
        "season": season,
        "mnth": month,
        "hr": hour,
        "weekday": weekday,
        "weathersit": weather
    }

    for category, value in category_values.items():

        encoded_column = f"{category}_{value}"

        if encoded_column in input_data.columns:
            input_data.loc[0, encoded_column] = 1

        elif category in input_data.columns:
            input_data.loc[0, category] = value

    # Make sure prediction uses the exact training columns
    input_data = input_data[feature_columns]

    # Predict
    prediction = float(model.predict(input_data)[0])

    # Rental count cannot be negative
    prediction = max(0, prediction)

    # 6. DISPLAY RESULT


    st.subheader("Prediction Result")

    st.metric(
        label="Estimated rentals for this hour",
        value=f"{prediction:.0f} bikes"
    )

    st.write("Selected conditions:")

    st.dataframe(
        pd.DataFrame({
            "Condition": [
                "Year",
                "Month",
                "Hour",
                "Weekday",
                "Season",
                "Weather",
                "Holiday",
                "Working day",
                "Temperature",
                "Feels-like temperature",
                "Humidity",
                "Wind speed"
            ],
            "Value": [
                year,
                month,
                hour,
                weekday,
                season,
                weather,
                holiday,
                workingday,
                temp,
                atemp,
                hum,
                windspeed
            ]
        }),
        hide_index=True
    )

    st.caption(
        "This is a model estimate, not a guaranteed future rental count."
    )


# 7. FOOTER

st.divider()
st.write(
    "Built with Python, Pandas, Scikit-learn and Streamlit."
)

