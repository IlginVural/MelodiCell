from cnnModel import EmotionModel
from fer import FER2013DataLoader
from real_time import RealTimeEmotionDetector

def train_and_save_model():
    data_loader = FER2013DataLoader('fer2013.csv')
    X_train, X_test, y_train, y_test = data_loader.load_data()

    model = EmotionModel.create_model()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=64)

    model.save('emotion_detection_model.h5')
    print("Model trained and saved successfully!")

def run_real_time_detection():
    detector = RealTimeEmotionDetector('emotion_detection_model.h5')
    detector.detect_emotions()

if __name__ == "__main__":
    # First, train and save the model
    # train_and_save_model()

    # Or if model already trained, directly run real-time detection
    run_real_time_detection()
