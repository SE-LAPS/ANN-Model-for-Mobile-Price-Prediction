import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
file_path = 'Mobile_Price_Classification.csv'
data = pd.read_csv(file_path)

# Separate the features and the target
X = data.drop('price_range', axis=1)
y = data['price_range']

# Split the data into training (75%) and testing (25%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the ANN model
model = Sequential([
    Dense(8, input_dim=20, activation='relu'),  # 1st hidden layer with 8 neurons
    Dense(4, activation='relu'),  # 2nd hidden layer with 4 neurons
    Dense(1, activation='sigmoid')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Save the model weights
model.save_weights('ann_model_weights.weights.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')