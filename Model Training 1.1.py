import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import time
from skimage.feature import hog
from skimage import io, color, exposure 
import matplotlib.pyplot as plt 


class EnhancedFluorescentImmunoassayAI:
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.svm = SVC(probability=True, random_state=42)
        
        self.model = VotingClassifier(
            estimators=[('rf', self.rf), ('gb', self.gb), ('svm', self.svm)],
            voting='soft'
        )

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, threshold = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        return threshold

    def extract_features(self, processed_image):
        mean_intensity = np.mean(processed_image)
        max_intensity = np.max(processed_image)
        signal_area = np.sum(processed_image > 0)
        
        # Additional features
        std_intensity = np.std(processed_image)
        median_intensity = np.median(processed_image)
        
        return [mean_intensity, max_intensity, signal_area, std_intensity, median_intensity]

    def train_model(self, image_paths, labels):
        features = []
        valid_labels = []
        for image_path, label in zip(image_paths, labels):
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    processed_image = self.preprocess_image(image)
                    features.append(self.extract_features(processed_image))
                    valid_labels.append(label)
                else:
                    print(f"Warning: Could not read image {image_path}")
            else:
                print(f"Warning: Image file not found: {image_path}")

        if not features:
            print("Error: No valid images found. Cannot train the model.")
            return

        X = np.array(features)
        y = np.array(valid_labels)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2f}")
        print(classification_report(y_test, y_pred))

    def predict(self, image_path):
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return None

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None

        processed_image = self.preprocess_image(image)
        features = self.extract_features(processed_image)
        prediction = self.model.predict([features])
        return prediction[0]

    def save_model(self, filename):
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        if os.path.exists(filename):
            self.model = joblib.load(filename)
            print(f"Model loaded from {filename}")
        else:
            print(f"Error: Model file not found: {filename}")

def main():
    ai_system = EnhancedFluorescentImmunoassayAI()
    model_filename = "enhanced_als_model.joblib"

    # Your existing image_paths and labels here
    image_paths = [
        
    ]
    labels = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1,1, 0, 1]  # 1 label as postive 0 as negative 

    while True:
        print("\n--- Enhanced Fluorescent Immunoassay AI System ---")
        print("1. Train model")
        print("2. Load model")
        print("3. Make prediction")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            ai_system.train_model(image_paths, labels)
            ai_system.save_model(model_filename)

        elif choice == '2':
            ai_system.load_model(model_filename)

        elif choice == '3':
            if not hasattr(ai_system.model, 'predict'):
                print("Error: No model loaded. Please train or load a model first.")
                continue

            image_path = input("Enter the path of the image to predict: ").strip()
            prediction = ai_system.predict(image_path)
            if prediction is not None:
                print(f"Prediction for {image_path}: {'Positive (ALS)' if prediction == 1 else 'Negative'}")

        elif choice == '4':
            print("Exiting the program. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

        time.sleep(1)

if __name__ == "__main__":
    main()
