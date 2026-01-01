from tensorflow import keras
from keras.models import load_model
import numpy as np


# Load model
import os



model = load_model('./house_price_model.keras')
# Load data 

from tensorflow.keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# new_data = [
#     1.23247, 0., 8.14, 0., 0.538, 6.142, 91.7,
#      3.9769, 4., 307., 21., 396.9, 18.72
# ]
new_data = train_data[0]
new_data = new_data.reshape(1, -1)


prediction = model.predict(new_data)
print(f"Predicted price: ${prediction[0][0]:.2f}")