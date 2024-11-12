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
        
        # Define confidence thresholds
        self.confidence_levels = {
            'very_high': 0.90,
            'high': 0.80,
            'moderate': 0.65,
            'low': 0.50
        }

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, threshold = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        return threshold, gray  # Return both for analysis

    def extract_features(self, processed_image, original_gray):
        mean_intensity = np.mean(processed_image)
        max_intensity = np.max(processed_image)
        signal_area = np.sum(processed_image > 0)
        std_intensity = np.std(processed_image)
        median_intensity = np.median(processed_image)
        
        # Additional signal quality metrics
        signal_to_noise = mean_intensity / std_intensity if std_intensity > 0 else 0
        intensity_ratio = max_intensity / (mean_intensity + 1e-6)
        
        return [mean_intensity, max_intensity, signal_area, std_intensity, 
                median_intensity, signal_to_noise, intensity_ratio]

    def get_confidence_level(self, probability):
        if probability >= self.confidence_levels['very_high']:
            return "Very High"
        elif probability >= self.confidence_levels['high']:
            return "High"
        elif probability >= self.confidence_levels['moderate']:
            return "Moderate"
        elif probability >= self.confidence_levels['low']:
            return "Low"
        else:
            return "Very Low"

    def analyze_signal_quality(self, features):
        signal_to_noise = features[5]
        intensity_ratio = features[6]
        
        if signal_to_noise > 5 and intensity_ratio > 2:
            return "Excellent"
        elif signal_to_noise > 3 and intensity_ratio > 1.5:
            return "Good"
        elif signal_to_noise > 2 and intensity_ratio > 1.2:
            return "Fair"
        else:
            return "Poor"

    def train_model(self, image_paths, labels):
        features = []
        valid_labels = []
        training_summary = {
            'processed_images': 0,
            'failed_images': 0,
            'quality_distribution': {'Excellent': 0, 'Good': 0, 'Fair': 0, 'Poor': 0}
        }

        for image_path, label in zip(image_paths, labels):
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    processed_image, gray_image = self.preprocess_image(image)
                    feature_vector = self.extract_features(processed_image, gray_image)
                    features.append(feature_vector)
                    valid_labels.append(label)
                    
                    # Track quality metrics
                    quality = self.analyze_signal_quality(feature_vector)
                    training_summary['quality_distribution'][quality] += 1
                    training_summary['processed_images'] += 1
                else:
                    print(f"Warning: Could not read image {image_path}")
                    training_summary['failed_images'] += 1
            else:
                print(f"Warning: Image file not found: {image_path}")
                training_summary['failed_images'] += 1

        if not features:
            print("Error: No valid images found. Cannot train the model.")
            return

        X = np.array(features)
        y = np.array(valid_labels)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        print("\nTraining model...")
        self.model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Print training summary
        print("\n=== Training Summary ===")
        print(f"Total images processed: {training_summary['processed_images']}")
        print(f"Failed images: {training_summary['failed_images']}")
        print("\nSignal Quality Distribution:")
        for quality, count in training_summary['quality_distribution'].items():
            percentage = (count / training_summary['processed_images']) * 100
            print(f"- {quality}: {count} ({percentage:.1f}%)")
        
        print(f"\nModel accuracy: {accuracy:.2f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Additional confidence analysis
        confidence_levels = [max(proba) for proba in y_pred_proba]
        print("\nConfidence Analysis:")
        print(f"Average prediction confidence: {np.mean(confidence_levels):.2f}")
        print(f"Minimum prediction confidence: {np.min(confidence_levels):.2f}")
        print(f"Maximum prediction confidence: {np.max(confidence_levels):.2f}")

    def predict(self, image_path):
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}

        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"Could not read image {image_path}"}

        processed_image, gray_image = self.preprocess_image(image)
        features = self.extract_features(processed_image, gray_image)
        
        # Get probability predictions
        probabilities = self.model.predict_proba([features])[0]
        prediction = self.model.predict([features])[0]
        
        # Calculate confidence and determine signal quality
        confidence_score = max(probabilities)
        confidence_level = self.get_confidence_level(confidence_score)
        signal_quality = self.analyze_signal_quality(features)
        
        # Prepare detailed analysis report
        result = {
            "prediction": "Positive" if prediction == 1 else "Negative",
            "confidence_score": f"{confidence_score:.2%}",
            "confidence_level": confidence_level,
            "signal_quality": signal_quality,
            "analysis": {
                "mean_intensity": features[0],
                "max_intensity": features[1],
                "signal_area": features[2],
                "signal_to_noise_ratio": features[5],
                "intensity_ratio": features[6]
            }
        }
        
        # Add interpretative comments
        result["interpretation"] = self._generate_interpretation(result)
        
        return result

    def _generate_interpretation(self, result):
        interpretation = []
        
        # Base conclusion
        if result["prediction"] == "Positive":
            base_message = "The sample shows indicators of being positive"
        else:
            base_message = "The sample shows indicators of being negative"
            
        # Confidence analysis
        if result["confidence_level"] in ["Very High", "High"]:
            confidence_message = f"with {result['confidence_level'].lower()} confidence ({result['confidence_score']})"
        else:
            confidence_message = f"but with {result['confidence_level'].lower()} confidence ({result['confidence_score']}), suggesting potential ambiguity"
            
        # Signal quality consideration
        quality_message = f"The signal quality is {result['signal_quality'].lower()}"
        if result["signal_quality"] in ["Poor", "Fair"]:
            quality_message += ", which may affect result reliability"
        
        # Combine messages
        interpretation.append(f"{base_message} {confidence_message}.")
        interpretation.append(quality_message + ".")
        
        # Additional recommendations if needed
        if result["confidence_level"] in ["Low", "Very Low"] or result["signal_quality"] in ["Poor"]:
            interpretation.append("Recommendation: Consider retesting the sample under optimal conditions.")
            
        return " ".join(interpretation)

    def save_model(self, filename):
        try:
            joblib.dump(self.model, filename)
            print(f"Model successfully saved to {filename}")
            
            # Save model metadata
            metadata = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'confidence_levels': self.confidence_levels
            }
            metadata_file = f"{os.path.splitext(filename)[0]}_metadata.json"
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            print(f"Model metadata saved to {metadata_file}")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")

    def load_model(self, filename):
        try:
            if os.path.exists(filename):
                self.model = joblib.load(filename)
                print(f"Model successfully loaded from {filename}")
                
                # Load model metadata if available
                metadata_file = f"{os.path.splitext(filename)[0]}_metadata.json"
                if os.path.exists(metadata_file):
                    import json
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    print(f"Model metadata loaded from {metadata_file}")
                    print(f"Model timestamp: {metadata.get('timestamp', 'Not available')}")
                return True
            else:
                print(f"Error: Model file not found: {filename}")
                return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

def main():
    ai_system = EnhancedFluorescentImmunoassayAI()
    model_filename = "enhanced_als_model.joblib"

    # Your image paths and labels here
    image_paths = [
        # Add your image paths here
    ]
    labels = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1]

    while True:
        print("\n=== Enhanced Fluorescent Immunoassay AI System ===")
        print("1. Train new model")
        print("2. Load existing model")
        print("3. Analyze sample")
        print("4. Batch analysis")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")

        if choice == '1':
            print("\nInitiating model training...")
            ai_system.train_model(image_paths, labels)
            save = input("\nWould you like to save the trained model? (y/n): ")
            if save.lower() == 'y':
                ai_system.save_model(model_filename)

        elif choice == '2':
            model_path = input("\nEnter model path (press Enter for default 'enhanced_als_model.joblib'): ").strip()
            if not model_path:
                model_path = model_filename
            ai_system.load_model(model_path)

        elif choice == '3':
            if not hasattr(ai_system.model, 'predict'):
                print("\nError: No model loaded. Please train or load a model first.")
                continue

            image_path = input("\nEnter the path of the image to analyze: ").strip()
            result = ai_system.predict(image_path)
            
            if "error" in result:
                print(f"\nError: {result['error']}")
                continue
                
            print("\n=== Analysis Results ===")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence Level: {result['confidence_level']} ({result['confidence_score']})")
            print(f"Signal Quality: {result['signal_quality']}")
            print("\nDetailed Analysis:")
            for key, value in result['analysis'].items():
                print(f"- {key.replace('_', ' ').title()}: {value:.2f}")
            print("\nInterpretation:")
            print(result['interpretation'])

        elif choice == '4':
            if not hasattr(ai_system.model, 'predict'):
                print("\nError: No model loaded. Please train or load a model first.")
                continue

            folder_path = input("\nEnter folder path containing images to analyze: ").strip()
            if not os.path.exists(folder_path):
                print("Error: Folder not found")
                continue

            print("\nProcessing batch analysis...")
            results = []
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    image_path = os.path.join(folder_path, filename)
                    result = ai_system.predict(image_path)
                    results.append((filename, result))

            print("\n=== Batch Analysis Results ===")
            for filename, result in results:
                if "error" in result:
                    print(f"\n{filename}: Error - {result['error']}")
                else:
                    print(f"\n{filename}:")
                    print(f"Prediction: {result['prediction']}")
                    print(f"Confidence: {result['confidence_level']} ({result['confidence_score']})")
                    print(f"Signal Quality: {result['signal_quality']}")

        elif choice == '5':
            print("\nExiting program. Goodbye!")
            break

        else:
            print("\nInvalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()
