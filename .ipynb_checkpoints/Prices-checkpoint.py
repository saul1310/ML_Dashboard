from tensorflow.keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
train_data.shape
test_data.shape
train_targets
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

# Preparing the data
# Normalizing the data

# Boston Housing Price Prediction - Regression Neural Network
# This implements a regression model (predicting continuous values, not categories)
# Task: Predict median house prices in Boston suburbs based on features like
#crime rate, number of rooms, property tax rate, etc.


# ============================================================================
# DATA NORMALIZATION
# ============================================================================


# Normalizing the data using standardization (z-score normalization)
# Formula: z = (x - mean) / std, Prevents features with large values from dominating

mean = train_data.mean(axis=0) # Calculate mean for each of the 13 features
train_data -= mean    # Subtract mean from training data
std = train_data.std(axis=0)  # Calculate standard deviation for each feature
train_data /= std  # Divide by standard deviation
test_data -= mean #Use the SAME mean and std from training data on test data, because,
# looking at the mean from the test data would be like peeking at info we wont have in production
test_data /= std

# What is standardization?
# Formula: z = (x - mean) / std
# Result: Data centered around 0 with standard deviation of 1
#Why are we using standardization this time?
#  - Better for features that can have outliers
#   - Works well with features that have different distributions
#   - Common practice in regression problems

#  After standardization:
#   - Most values fall between -3 and +3
#   - Mean = 0, Standard deviation = 1
#   - Keeps relative distances between data points




# ============================================================================
# MODEL BUILDING
# ============================================================================

"""
    Build and compile a neural network for regression.
    
    Returns:
        A compiled Keras model ready for training
    
    Role: Creates the architecture for our regression neural network.
    
    Architecture:
        Input layer: 13 features (implicit, not defined)
        Hidden layer 1: 64 neurons with ReLU activation
        Hidden layer 2: 64 neurons with ReLU activation
        Output layer: 1 neuron with NO activation (linear output)
    
    Why this architecture?
        - 64 neurons is a good starting point for medium-complexity problems
        - 2 hidden layers allows the network to learn complex patterns
        - ReLU activation (max(0, x)) is faster and works better than sigmoid
          for hidden layers in most modern networks
        - Output layer has 1 neuron because we're predicting 1 number (house price)
        - NO activation on output = can predict any real number (not just 0-1)- a sigmoid activation
        on the last output layer would put the number between 0 and 1
    
    Compilation settings:

    ReLU (Rectified Linear Unit) is a non-linear 
    activation function used in neural networks that outputs the input 
    if it is positive and zero otherwise

        - optimizer="rmsprop": An adaptive learning rate algorithm
          (automatically adjusts learning speed during training)
        
        - loss="mse": Mean Squared Error
          Formula: average of (predicted - actual)²
          Penalizes large errors heavily (squaring makes big errors much worse)
          Example: If predicted=$30k, actual=$25k → error = (30-25)² = 25
        
        - metrics=["mae"]: Mean Absolute Error (for monitoring only)
          Formula: average of |predicted - actual|
          Easier to interpret than MSE (in same units as target)
          Example: If predicted=$30k, actual=$25k → error = |30-25| = 5
          MAE tells us on average, we're off by $5,000
    
    """
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model




# ============================================================================
# K-FOLD CROSS-VALIDATION (FIRST PASS - QUICK EVALUATION)
# ============================================================================

# K-fold validation
# Why K-fold for this dataset?
# - We only have 404 training samples which is not a lot
# - A single train/test split might not be representative
# - K-fold gives us more reliable performance estimates

k = 4 # Split data into 4 folds
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

# How K-fold works with k=4:
# Fold 0: samples 0-100 for validation,   samples 101-403 for training
# Fold 1: samples 101-201 for validation, samples 0-100 & 202-403 for training
# Fold 2: samples 202-302 for validation, samples 0-201 & 303-403 for training
# Fold 3: samples 303-403 for validation, samples 0-302 for training


for i in range(k):
    print(f"Processing fold #{i}")
    # STEP 1: Create validation set for this fold
    # Select a contiguous chunk of data for validation
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
     # STEP 2: Create training set (everything but the validation chunk we just nabbed)
    # Use np.concatenate to combine data before and after the validation chunk
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    # STEP 3: Build a fresh model for this fold
    # Important: Each fold gets a different model with random weights
    # This ensures fair comparison (not influenced by previous training we just did)
    model = build_model()

    # STEP 4: Train the model
    # batch_size=16: Process 16 samples at a time before updating weights
    # verbose=0: Silent training (no progress output)
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=16, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

# Results after training 4 models
all_scores
#  Average MAE across all folds
 # This is our best estimate of model performance
np.mean(all_scores)

# ============================================================================
# K-FOLD CROSS-VALIDATION (SECOND PASS - DETAILED HISTORY)
# ============================================================================

# Now we run K-fold again with a lot epochs and SAVE the history
# This lets us plot how validation error changes over time
# Saving the validation logs at each fold

num_epochs = 500 # Train longer to see full learning curve
all_mae_histories = []
#  This loop is almost identical to the previous one, but:
# 1. We train for 500 epochs instead of 100
# 2. We pass validation_data to model.fit() to get validation metrics during training
# 3. We save the history of validation MAE values
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=16, verbose=0)
    mae_history = history.history["val_mae"]
    all_mae_histories.append(mae_history)

# ============================================================================
# AVERAGING VALIDATION SCORES ACROSS FOLDS
# ============================================================================
# Building the history of successive mean K-fold validation scores
# Goal: For each epoch, average the MAE across all 4 folds
# This smooths out the noise and gives us a reliable learning curve

average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]


# explanation
# For epoch 0: take MAE from fold0[0], fold1[0], fold2[0], fold3[0] → average them
# For epoch 1: take MAE from fold0[1], fold1[1], fold2[1], fold3[1] → average them
# ... and so on for all 500 epochs
#
# Result: average_mae_history = [3.5, 3.2, 2.9, 2.7, 2.5, ..., 2.1, 2.1, 2.1]
#         Shows how MAE improves, meaning the score decreases as training progresses

# ============================================================================
# PLOTTING VALIDATION SCORES
# ============================================================================


plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()

# Plotting validation scores, excluding the first 10 data points

truncated_mae_history = average_mae_history[10:]
plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()

# Training the final model
model = build_model()
model.fit(train_data, train_targets,
          epochs=130, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
test_mae_score
# Generating predictions on new data
predictions = model.predict(test_data)
predictions[0]