import numpy as np
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE # type: ignore
from skimage.feature import hog
import joblib
import os

class EnhancedFluorescentImmunoassayAI:
    def __init__(self):
        self.rf = RandomForestClassifier(random_state=42)
        self.gb = GradientBoostingClassifier(random_state=42)
        self.svm = SVC(probability=True, random_state=42)
        self.scaler = StandardScaler()
        
    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        _, threshold = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return threshold

    def extract_features(self, processed_image):
        hog_features = hog(processed_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        
        mean_intensity = np.mean(processed_image)
        max_intensity = np.max(processed_image)
        signal_area = np.sum(processed_image > 0)
        std_intensity = np.std(processed_image)
        median_intensity = np.median(processed_image)
        
        return np.concatenate(([mean_intensity, max_intensity, signal_area, std_intensity, median_intensity], hog_features))

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

        # Apply SMOTE for balancing classes
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_resampled)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

        # Hyperparameter tuning
        rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
        gb_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
        svm_params = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}

        rf_grid = GridSearchCV(self.rf, rf_params, cv=5, n_jobs=-1)
        gb_grid = GridSearchCV(self.gb, gb_params, cv=5, n_jobs=-1)
        svm_grid = GridSearchCV(self.svm, svm_params, cv=5, n_jobs=-1)

        rf_grid.fit(X_train, y_train)
        gb_grid.fit(X_train, y_train)
        svm_grid.fit(X_train, y_train)

        self.rf = rf_grid.best_estimator_
        self.gb = gb_grid.best_estimator_
        self.svm = svm_grid.best_estimator_

        # Ensemble voting
        self.model = VotingClassifier(
            estimators=[('rf', self.rf), ('gb', self.gb), ('svm', self.svm)],
            voting='soft'
        )

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
        scaled_features = self.scaler.transform([features])
        prediction = self.model.predict(scaled_features)
        return prediction[0]

    def save_model(self, filename):
        joblib.dump({'model': self.model, 'scaler': self.scaler}, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        if os.path.exists(filename):
            loaded = joblib.load(filename)
            self.model = loaded['model']
            self.scaler = loaded['scaler']
            print(f"Model loaded from {filename}")
        else:
            print(f"Error: Model file not found: {filename}")

def main():
    ai_system = EnhancedFluorescentImmunoassayAI()
    model_filename = "enhanced_als_model.joblib"

    # Your existing image_paths and labels here
    image_paths = [
        #r"C:\Users\lavai\Downloads\strip-test-lateral-flow-assay-for-evaluation-of-an-antibody-europium-particle-conjugate-photographic-image.jpg",  #Positive
        #r"C:\Users\lavai\Downloads\A-LFIA-with-different-running-buffers-From-left-to-right-strips-with.png", #negative
        #r"c:\Users\lavai\Downloads\1-s2.0-S0026265X23005507-gr3.jpg", #positive
        #r"c:\Users\lavai\Downloads\41551_2020_655_Fig3_HTML.png", #negative
        #r"c:\Users\lavai\Downloads\fbioe-10-1042926-g004.jpg", #Positive
        #r"c:\Users\lavai\Downloads\The-limit-of-detection-and-cross-reactivity-of-the-lateral-flow-assay-a-b-The-limit-of_Q320.jpg", #negative
        #r"c:\Users\lavai\Downloads\Fluorescent-images-of-the-LFA-acquired-under-UV-light-and-the-calibration-curves-of_Q320.jpg", #negative
        #r"c:\Users\lavai\Downloads\Fluorescent-Universal-LFA.jpg", #Negative
        #r"c:\Users\lavai\Downloads\Capture-204x300.png", #negative
        #r"c:\Users\lavai\Downloads\WebPageImage2024-04-17-135049.png", #negative
        #r"c:\Users\lavai\Downloads\Standard-curve-and-linearity-of-the-QD-based-LFIA-strip-n10.png", #negative
        #r"c:\Users\lavai\Downloads\Screenshot-2024-06-10-103538.png", #positive

        # Next files will only be blank or negative tests. 
        r"c:\Users\lavai\Downloads\Screenshot 2024-09-14 174731.png", #negative
        r"c:\Users\lavai\OneDrive\Pictures\Screenshots\Screenshot 2024-09-14 235649.png", #0
        r"c:\Users\lavai\Downloads\Staphylococcus-aureus-Panton-Valentine-leukocidin-Lateral-Flow-Immunoassay-A-positive_Q320.jpg", # 0 
        r"c:\Users\lavai\Downloads\Screenshot 2024-10-11 213904.png", #1
        r"c:\Users\lavai\OneDrive\Pictures\Screenshots\Screenshot 2024-10-11 213452.png",#1
        r"c:\Users\lavai\OneDrive\Pictures\Screenshots\Screenshot 2024-10-11 213216.png", #1
        r"c:\Users\lavai\OneDrive\Pictures\Screenshots\Screenshot 2024-10-12 164915.png", #1
        r"c:\Users\lavai\OneDrive\Pictures\Screenshots\Screenshot 2024-10-12 164858.png",#1
        r"C:\Users\lavai\OneDrive\Pictures\Screenshots\Screenshot 2024-10-12 170010.png",#0
        r"c:\Users\lavai\OneDrive\Pictures\Screenshots\Screenshot 2024-10-12 170106.png", #0
        r"c:\Users\lavai\OneDrive\Pictures\Screenshots\Screenshot 2024-10-12 171000.png",#0
        r"C:\Users\lavai\OneDrive\Pictures\Screenshots\Screenshot 2024-10-12 171914.png", #0
        r"c:\Users\lavai\OneDrive\Pictures\Screenshots\Screenshot 2024-10-12 172330.png", #0
        r"C:\Users\lavai\OneDrive\Pictures\Screenshots\Screenshot 2024-10-13 171201.png", #1
        r"c:\Users\lavai\OneDrive\Pictures\Screenshots\Screenshot 2024-10-13 172006.png",#1
        r"C:\Users\lavai\Downloads\IMG_5743.jpg",#1
        r"C:\Users\lavai\OneDrive\Pictures\Screenshots\Screenshot 2024-10-13 172127.png", #0
        r"C:\Users\lavai\OneDrive\Pictures\Screenshots\Screenshot 2024-10-11 213904.png", #1
        r"C:\Users\lavai\OneDrive\Pictures\Screenshots\Screenshot 2024-10-13 165410.png", #1 
        r"C:\Users\lavai\OneDrive\Pictures\Screenshots\Screenshot 2024-10-13 173229.png", #0
        r"C:\Users\lavai\OneDrive\Pictures\Screenshots\Screenshot 2024-10-13 173308.png", #1
        r"C:\Users\lavai\OneDrive\Pictures\Screenshots\Screenshot 2024-10-13 173433.png", #1
        r"C:\Users\lavai\OneDrive\Pictures\Screenshots\Screenshot 2024-10-13 173504.png", # 0
        r"C:\Users\lavai\OneDrive\Pictures\Screenshots\Screenshot 2024-10-13 173955.png", # 1
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