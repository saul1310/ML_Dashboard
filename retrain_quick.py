import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import boston_housing

# Quick retrain script: compute mean/std, normalize, train final model, save
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

# Build model
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(train_data.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model

model = build_model()

# Train
model.fit(train_data, train_targets, epochs=130, batch_size=16, verbose=1)

# Evaluate
test_mse, test_mae = model.evaluate(test_data, test_targets, verbose=1)
print(f"Test MAE: {test_mae}")

# Save model and normalization params
model.save('house_price_model.keras')
np.save('mean.npy', mean)
np.save('std.npy', std)
print('Saved model and mean/std.')
