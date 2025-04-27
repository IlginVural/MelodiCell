import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class FER2013DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        data = pd.read_csv(self.filepath)

        X = []
        y = []

        for index, row in data.iterrows():
            X.append(np.array(row['pixels'].split(), dtype='float32').reshape(48, 48, 1))
            y.append(row['emotion'])

        X = np.array(X) / 255.0
        y = to_categorical(np.array(y), num_classes=7)

        return train_test_split(X, y, test_size=0.2, random_state=42)
    
#imports needed to be fixed 