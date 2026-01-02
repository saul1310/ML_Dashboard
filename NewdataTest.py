from tensorflow import keras
from keras.models import load_model
import numpy as np


# Load model
import os



model = load_model('./house_price_model.keras')
# Load data 

from tensorflow.keras.datasets import boston_housing

mean = np.load('./mean.npy')
std = np.load('./std.npy')

# Load fresh data
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# Get sample and normalize it
new_data = test_data[0].reshape(1, -1)
new_data_normalized = (new_data - mean) / std  # <-- NORMALIZE IT!

# Predict
prediction = model.predict(new_data_normalized)
print(f"Predicted price: ${prediction[0][0] * 1000:.2f}")  # Multiply by 1000!
print(f"Expected price: ${test_targets[0] * 1000:.2f}")