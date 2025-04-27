import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('fer2013.csv')

# Preprocess the data
X = []
y = []
for index, row in data.iterrows():
    X.append(np.array(row['pixels'].split(), dtype='float32').reshape(48, 48, 1))
    y.append(row['emotion'])

X = np.array(X) / 255.0  # Normalize pixel values
y = to_categorical(np.array(y), num_classes=7)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
