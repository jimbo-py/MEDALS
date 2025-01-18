import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import time
import json

class EnhancedFluorescentImmunoassayAI:
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.svm = SVC(probability=True, random_state=42)
        
        self.model = VotingClassifier(
            estimators=[('rf', self.rf), ('gb', self.gb), ('svm', self.svm)],
            voting='soft'
        )
        
        self.confidence_levels = {
            'very_high': 0.90,
            'high': 0.80,
            'moderate': 0.65,
            'low': 0.50
        }
        
        self.is_fitted = False

    def load_model(self, filename):
        """
        Load a previously trained model from a file.
        
        Parameters:
        filename (str): Path to the saved model file
        
        Returns:
        bool: True if model loaded successfully, False otherwise
        """
        try:
            self.model = joblib.load(filename)
            self.is_fitted = True
            print(f"Model successfully loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def train_model(self, image_paths, labels):
        """
        Train the model using a list of image paths and their corresponding labels.
        
        Parameters:
        image_paths: list of str
            List of paths to training images
        labels: list of int
            List of labels (0 or 1) corresponding to each image
            
        Returns:
        dict: Training summary
        list: List of invalid image paths
        """
        if len(image_paths) != len(labels):
            raise ValueError("Number of images and labels must match")
            
        features_list = []
        valid_labels = []
        invalid_images = []
        
        print("Processing training images...")
        
        for idx, (img_path, label) in enumerate(zip(image_paths, labels)):
            try:
                # Read and process image
                image = cv2.imread(img_path)
                if image is None:
                    raise ValueError(f"Could not read image: {img_path}")
                
                # Extract features
                processed_image, gray_image = self.preprocess_image(image)
                features = self.extract_features(processed_image, gray_image)
                
                features_list.append(features)
                valid_labels.append(label)
                print(f"Successfully processed image {idx + 1}/{len(image_paths)}: {os.path.basename(img_path)}")
                
            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
                invalid_images.append(img_path)
                continue
        
        if not features_list:
            raise ValueError("No valid images to train on")
        
        # Convert to numpy arrays
        X_train = np.array(features_list)
        y_train = np.array(valid_labels)
        
        print("\nTraining model...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Calculate training metrics
        y_pred = self.model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        
        training_summary = {
            "total_images": len(image_paths),
            "valid_images": len(features_list),
            "invalid_images": len(invalid_images),
            "training_accuracy": accuracy,
            "feature_dimensions": len(features_list[0])
        }
        
        print("\nTraining Summary:")
        print(f"Total Images: {training_summary['total_images']}")
        print(f"Valid Images: {training_summary['valid_images']}")
        print(f"Invalid Images: {training_summary['invalid_images']}")
        print(f"Training Accuracy: {training_summary['training_accuracy']:.2%}")
        
        return training_summary, invalid_images

    def save_model(self, filename):
        """Save the trained model to a file."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet")
        
        try:
            joblib.dump(self.model, filename)
            print(f"Model successfully saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False

    def predict(self, image):
        """
        Make a prediction on a single image.
        
        Parameters:
        image: numpy.ndarray
            The input image to analyze
            
        Returns:
        dict: Prediction results and analysis
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet")
            
        try:
            # Preprocess the image
            processed_image, gray_image = self.preprocess_image(image)
            features = self.extract_features(processed_image, gray_image)
            
            # Make prediction
            features_array = np.array([features])
            prediction_proba = self.model.predict_proba(features_array)[0]
            prediction = "Positive" if prediction_proba[1] > 0.5 else "Negative"
            confidence_score = max(prediction_proba)
            
            # Determine confidence level
            confidence_level = "low"
            for level, threshold in sorted(self.confidence_levels.items(), 
                                        key=lambda x: x[1], reverse=True):
                if confidence_score >= threshold:
                    confidence_level = level
                    break
            
            # Calculate signal quality metrics
            signal_quality = "Good" if features[5] > 2.0 else "Poor"
            
            # Create analysis dictionary
            analysis = {
                "mean_intensity": features[0],
                "max_intensity": features[1],
                "signal_area": features[2],
                "std_intensity": features[3],
                "median_intensity": features[4],
                "signal_to_noise": features[5],
                "intensity_ratio": features[6]
            }
            
            # Generate interpretation
            interpretation = self._generate_interpretation(prediction, confidence_level, 
                                                        signal_quality, analysis)
            
            return {
                "prediction": prediction,
                "confidence_score": confidence_score,
                "confidence_level": confidence_level,
                "signal_quality": signal_quality,
                "analysis": analysis,
                "interpretation": interpretation
            }
            
        except Exception as e:
            raise ValueError(f"Error during prediction: {str(e)}")

    def preprocess_image(self, image):
        """Preprocess the input image for feature extraction."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, threshold = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        return threshold, gray

    def extract_features(self, processed_image, original_gray):
        """Extract features from the processed image."""
        mean_intensity = np.mean(processed_image)
        max_intensity = np.max(processed_image)
        signal_area = np.sum(processed_image > 0)
        std_intensity = np.std(processed_image)
        median_intensity = np.median(processed_image)
        
        signal_to_noise = mean_intensity / std_intensity if std_intensity > 0 else 0
        intensity_ratio = max_intensity / (mean_intensity + 1e-6)
        
        return [mean_intensity, max_intensity, signal_area, std_intensity, 
                median_intensity, signal_to_noise, intensity_ratio]

    def _generate_interpretation(self, prediction, confidence_level, signal_quality, analysis):
        """Generate a human-readable interpretation of the results."""
        interpretation = []
        
        # Basic result interpretation
        if prediction == "Positive":
            interpretation.append("The analysis indicates a positive result")
        else:
            interpretation.append("The analysis indicates a negative result")
            
        # Confidence level interpretation
        confidence_messages = {
            "very_high": "with very high confidence",
            "high": "with high confidence",
            "moderate": "with moderate confidence",
            "low": "with low confidence"
        }
        interpretation[0] += f" {confidence_messages[confidence_level]}."
        
        # Signal quality interpretation
        if signal_quality == "Good":
            interpretation.append("The signal quality is good, suggesting reliable results.")
        else:
            interpretation.append("The signal quality is poor, which may affect result reliability.")
        
        # Feature analysis
        if analysis["signal_to_noise"] > 2.0:
            interpretation.append("The signal-to-noise ratio is favorable.")
        else:
            interpretation.append("The signal-to-noise ratio is lower than optimal.")
            
        if analysis["intensity_ratio"] > 1.5:
            interpretation.append("Strong signal intensity detected.")
        else:
            interpretation.append("Moderate to weak signal intensity detected.")
        
        return " ".join(interpretation)

class EnhancedALSPredictionUI:
    def __init__(self, master):
        self.master = master
        master.title("Enhanced ALS Prediction Tool")
        master.geometry("600x800")

        # Initialize AI system and load model
        self.ai_system = EnhancedFluorescentImmunoassayAI()
        self.load_or_train_model()

        self.main_frame = tk.Frame(master)
        self.main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        self.create_clinical_section()
        self.create_image_section()
        self.create_results_section()

        self.image_path = None

    def load_or_train_model(self):
        """Initialize the model by either loading or training it"""
        model_path = "trained_immunoassay_model.joblib"
        
        if os.path.exists(model_path):
            success = self.ai_system.load_model(model_path)
            if not success:
                self.train_new_model()
        else:
            self.train_new_model()

    def train_new_model(self):
        """Train a new model with the sample data"""
        try:
            image_paths = [
               r"/Users/alexvaisman/Desktop/Screenshot 2025-01-17 at 6.56.22 PM.png", 
               r"/Users/alexvaisman/Desktop/Screenshot 2025-01-17 at 6.56.30 PM.png", 
            ]
            
            labels = [0, 1]
            
            # Train the model
            training_summary, invalid_images = self.ai_system.train_model(image_paths, labels)
            
            # Save the trained model
            self.ai_system.save_model("trained_immunoassay_model.joblib")
            
            messagebox.showinfo("Success", "New model trained successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train new model: {str(e)}")
            self.master.destroy()

    def create_clinical_section(self):
        clinical_frame = tk.LabelFrame(self.main_frame, text="Clinical Information", padx=10, pady=10)
        clinical_frame.pack(fill=tk.X, pady=10)

        tk.Label(clinical_frame, text="Age:").pack()
        self.age_entry = tk.Entry(clinical_frame)
        self.age_entry.pack()

        tk.Label(clinical_frame, text="Sex:").pack()
        self.sex_var = tk.StringVar(value="Male")
        tk.Radiobutton(clinical_frame, text="Male", variable=self.sex_var, value="Male").pack()
        tk.Radiobutton(clinical_frame, text="Female", variable=self.sex_var, value="Female").pack()

        tk.Label(clinical_frame, text="ALSFRS-R Score (0-48):").pack()
        self.alsfrs_entry = tk.Entry(clinical_frame)
        self.alsfrs_entry.pack()
        
        tk.Label(clinical_frame, text="Background:").pack()
        self.eth_var = tk.StringVar(value="European")
        tk.Radiobutton(clinical_frame, text="White/European", variable=self.eth_var, value="White/European").pack()
        tk.Radiobutton(clinical_frame, text="Black/African", variable=self.eth_var, value="Black/African").pack()
        tk.Radiobutton(clinical_frame, text="Asian", variable=self.eth_var, value="Asian").pack()
        tk.Radiobutton(clinical_frame, text="Southasian", variable=self.eth_var, value="Southasian").pack()
        tk.Radiobutton(clinical_frame, text="Middle Eastern", variable=self.eth_var, value="Middle Eastern").pack()

    def create_image_section(self):
        image_frame = tk.LabelFrame(self.main_frame, text="Image Analysis", padx=10, pady=10)
        image_frame.pack(fill=tk.X, pady=10)

        self.upload_button = tk.Button(image_frame, text="Upload Fluorescent Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.image_status = tk.Label(image_frame, text="No image uploaded", fg="red")
        self.image_status.pack()

    def create_results_section(self):
        # Create results frame with proper configuration
        results_frame = tk.LabelFrame(self.main_frame, text="Results", padx=10, pady=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_rowconfigure(2, weight=1)  # Make row with text widget expandable

        # Add NCS input fields (row 0)
        self.create_ncs_inputs(results_frame)

        # Calculate button (row 1)
        self.calculate_button = tk.Button(results_frame, text="Calculate", command=self.calculate_als_chance)
        self.calculate_button.pack(pady=10)

        # Create a frame for the text widget with scrollbar (row 2)
        text_frame = tk.Frame(results_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add scrollbar
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure text widget with scrollbar
        self.result_text = tk.Text(text_frame, height=20, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.result_text.yview)

        # Save button (row 3)
        self.save_button = tk.Button(results_frame, text="Save Results", command=self.save_results)
        self.save_button.pack(pady=5)



    def create_ncs_inputs(self, parent_frame):
        """Create NCS input fields"""
        ncs_frame = tk.LabelFrame(parent_frame, text="Nerve Conduction Study (NCS)", padx=10, pady=10)
        ncs_frame.pack(fill=tk.X, pady=10)

        tk.Label(ncs_frame, text="Motor Nerve Conduction Velocity (m/s):").pack()
        self.motor_ncv_entry = tk.Entry(ncs_frame)
        self.motor_ncv_entry.pack()

        tk.Label(ncs_frame, text="Sensory Nerve Conduction Velocity (m/s):").pack()
        self.sensory_ncv_entry = tk.Entry(ncs_frame)
        self.sensory_ncv_entry.pack()

        tk.Label(ncs_frame, text="Compound Muscle Action Potential (mV):").pack()
        self.cmap_entry = tk.Entry(ncs_frame)
        self.cmap_entry.pack()

        tk.Label(ncs_frame, text="Sensory Nerve Action Potential (μV):").pack()
        self.snap_entry = tk.Entry(ncs_frame)
        self.snap_entry.pack()

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff")])
        if self.image_path:
            try:
                # Load and verify the image
                image = cv2.imread(self.image_path)
                if image is None:
                    raise ValueError("Unable to read the image file")
                
                self.image_status.config(text=f"Image loaded: {os.path.basename(self.image_path)}", fg="green")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.image_path = None
                self.image_status.config(text="Error loading image", fg="red")

    def validate_inputs(self):
        """Validate all user inputs before processing."""
        try:
            age = int(self.age_entry.get())
            if not (0 <= age <= 120):
                raise ValueError("Age must be between 0 and 120")
            
            alsfrs = float(self.alsfrs_entry.get())
            if not (0 <= alsfrs <= 48):
                raise ValueError("ALSFRS-R score must be between 0 and 48")
                
            if not self.image_path:
                raise ValueError("Please upload an image first")
                
            return True
            
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return False

    def validate_ncs_inputs(self):
        """Validate NCS inputs."""
        try:
            motor_ncv = float(self.motor_ncv_entry.get())
            sensory_ncv = float(self.sensory_ncv_entry.get())
            cmap = float(self.cmap_entry.get())
            snap = float(self.snap_entry.get())

            # Apply basic validation
            if motor_ncv <= 0 or sensory_ncv <= 0 or cmap <= 0 or snap <= 0:
                raise ValueError("All NCS values must be positive.")
            
            return {
                "motor_ncv": motor_ncv,
                "sensory_ncv": sensory_ncv,
                "cmap": cmap,
                "snap": snap,
            }
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return None

    def calculate_als_chance(self):
        """Process the image, clinical data, and NCS data for ALS prediction."""
        if not self.validate_inputs():
            return

        ncs_data = self.validate_ncs_inputs()
        if not ncs_data:
            return

        try:
            # Clear previous results
            self.result_text.delete(1.0, tk.END)
            self.result_text.update()

            # Load and analyze image
            image = cv2.imread(self.image_path)
            if image is None:
                raise ValueError("Failed to read the image")

            # Get prediction from AI system
            prediction_results = self.ai_system.predict(image)

            # Integrate NCS data into the analysis
            self.integrate_ncs_analysis(prediction_results, ncs_data)

            # Format results
            self.display_results(prediction_results)

        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error during analysis: {str(e)}")

    def integrate_ncs_analysis(self, prediction_results, ncs_data):
        """Integrate NCS data into the prediction results."""
        ncs_analysis = []

        # Motor NCV interpretation
        if ncs_data["motor_ncv"] < 40:
            ncs_analysis.append("Motor Nerve Conduction Velocity is significantly reduced, indicating possible motor neuropathy.")
        else:
            ncs_analysis.append("Motor Nerve Conduction Velocity is within the normal range.")

        # Sensory NCV interpretation
        if ncs_data["sensory_ncv"] < 45:
            ncs_analysis.append("Sensory Nerve Conduction Velocity is reduced, indicating potential sensory neuropathy.")
        else:
            ncs_analysis.append("Sensory Nerve Conduction Velocity is normal.")

        # CMAP interpretation
        if ncs_data["cmap"] < 3.0:
            ncs_analysis.append("Compound Muscle Action Potential is reduced, indicating possible muscle or motor neuron involvement.")
        else:
            ncs_analysis.append("Compound Muscle Action Potential is normal.")

        # SNAP interpretation
        if ncs_data["snap"] < 15.0:
            ncs_analysis.append("Sensory Nerve Action Potential is reduced, suggesting potential sensory nerve dysfunction.")
        else:
            ncs_analysis.append("Sensory Nerve Action Potential is normal.")

        # Append NCS analysis to the results
        prediction_results["ncs_analysis"] = ncs_analysis

    def display_results(self, prediction_results):
        """Display the analysis results, including NCS analysis, in a formatted way."""
    # Clinical Information Summary
        self.result_text.insert(tk.END, "CLINICAL INFORMATION\n")
        self.result_text.insert(tk.END, "-" * 50 + "\n")
        self.result_text.insert(tk.END, f"Age: {self.age_entry.get()} years\n")
        self.result_text.insert(tk.END, f"Sex: {self.sex_var.get()}\n")
        self.result_text.insert(tk.END, f"Background: {self.eth_var.get()}\n")
        self.result_text.insert(tk.END, f"ALSFRS-R Score: {self.alsfrs_entry.get()}\n")
        self.result_text.insert(tk.END, "\n")  # Added extra space for better readability

    # NCS Results
        self.result_text.insert(tk.END, "NERVE CONDUCTION STUDY RESULTS\n")
        self.result_text.insert(tk.END, "-" * 50 + "\n")
        self.result_text.insert(tk.END, f"Motor Nerve Conduction Velocity (m/s): {self.motor_ncv_entry.get()}\n")
        self.result_text.insert(tk.END, f"Sensory Nerve Conduction Velocity (m/s): {self.sensory_ncv_entry.get()}\n")
        self.result_text.insert(tk.END, f"Compound Muscle Action Potential (mV): {self.cmap_entry.get()}\n")
        self.result_text.insert(tk.END, f"Sensory Nerve Action Potential (μV): {self.snap_entry.get()}\n")
        self.result_text.insert(tk.END, "\n")  # Added extra space for better readability

    # NCS Analysis
        self.result_text.insert(tk.END, "NERVE CONDUCTION STUDY ANALYSIS\n")
        self.result_text.insert(tk.END, "-" * 50 + "\n")
        for line in prediction_results["ncs_analysis"]:
            self.result_text.insert(tk.END, f"{line}\n")
        self.result_text.insert(tk.END, "\n")  # Added extra space for better readability

    # Analysis Results
        self.result_text.insert(tk.END, "ANALYSIS RESULTS\n")
        self.result_text.insert(tk.END, "-" * 50 + "\n")
        self.result_text.insert(tk.END, f"Prediction: {prediction_results['prediction']}\n")
        self.result_text.insert(tk.END, f"Confidence Level: {prediction_results['confidence_level'].replace('_', ' ').title()}\n")
        self.result_text.insert(tk.END, f"Confidence Score: {prediction_results['confidence_score']:.2%}\n")
        self.result_text.insert(tk.END, f"Signal Quality: {prediction_results['signal_quality']}\n")
        self.result_text.insert(tk.END, "\n")  # Added extra space for better readability

    # Detailed Analysis
        self.result_text.insert(tk.END, "DETAILED ANALYSIS\n")
        self.result_text.insert(tk.END, "-" * 50 + "\n")
        analysis = prediction_results['analysis']
        self.result_text.insert(tk.END, f"Mean Intensity: {analysis['mean_intensity']:.2f}\n")
        self.result_text.insert(tk.END, f"Maximum Intensity: {analysis['max_intensity']:.2f}\n")
        self.result_text.insert(tk.END, f"Signal Area: {analysis['signal_area']:.2f}\n")
        self.result_text.insert(tk.END, f"Signal-to-Noise Ratio: {analysis['signal_to_noise']:.2f}\n")
        self.result_text.insert(tk.END, f"Intensity Ratio: {analysis['intensity_ratio']:.2f}\n")
        self.result_text.insert(tk.END, "\n")  # Added extra space for better readability

    # Interpretation
        self.result_text.insert(tk.END, "INTERPRETATION\n")
        self.result_text.insert(tk.END, "-" * 50 + "\n")
        self.result_text.insert(tk.END, f"{prediction_results['interpretation']}\n")
        self.result_text.insert(tk.END, "\n")  # Added extra space for better readability

    # Add timestamp
        self.result_text.insert(tk.END, f"\nAnalysis completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


    def save_results(self):
        """Save the analysis results to a file."""
        if not self.result_text.get(1.0, tk.END).strip():
            messagebox.showwarning("Warning", "No results to save")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(self.result_text.get(1.0, tk.END))
                messagebox.showinfo("Success", "Results saved successfully")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

def main():
    root = tk.Tk()
    app = EnhancedALSPredictionUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
