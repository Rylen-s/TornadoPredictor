#The dataset has been augmented using python due to a lack of time, but we can make this model work a lot better with real life data with good research
import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Number of records
n_records = 68000

# Define ranges for the variables based on realistic meteorological data
cape_range = (0, 5000)           # J/kg
wind_shear_range = (0, 30)       # m/s
humidity_range = (30, 100)       # %
pressure_range = (950, 1050)     # hPa
temperature_range = (-10, 40)    # °C

# Generate random data within these ranges
cape = np.random.uniform(cape_range[0], cape_range[1], n_records)
wind_shear = np.random.uniform(wind_shear_range[0], wind_shear_range[1], n_records)
humidity = np.random.uniform(humidity_range[0], humidity_range[1], n_records)
pressure = np.random.uniform(pressure_range[0], pressure_range[1], n_records)
temperature = np.random.uniform(temperature_range[0], temperature_range[1], n_records)

# Generate tornado occurrence (binary) based on simplified assumptions
# Higher CAPE, higher wind shear, and higher humidity increase the likelihood of tornadoes
tornado_probability = (
    0.0002 * cape + 
    0.02 * wind_shear + 
    0.01 * (humidity - 50) + 
    0.0001 * (1000 - np.abs(1000 - pressure)) + 
    0.01 * (temperature - 20)
)
tornado_probability = np.clip(tornado_probability, 0, 1)
tornado = np.random.binomial(1, tornado_probability)

# Create DataFrame
data = pd.DataFrame({
    "CAPE (J/kg)": cape,
    "Wind Shear (m/s)": wind_shear,
    "Humidity (%)": humidity,
    "Pressure (hPa)": pressure,
    "Temperature (°C)": temperature,
    "Tornado": ["Yes" if t else "No" for t in tornado]
})

data.head()
