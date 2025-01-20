import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Union
from pathlib import Path
import base64
import io

class FluorescentImmunoassayAnalyzer:
    def load_image(self, image_input: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
        """
        Load image from various input types: file path, bytes, or numpy array
        """
        if isinstance(image_input, (str, Path)):
            # Load from file path
            image = cv2.imread(str(image_input))
            if image is None:
                raise ValueError(f"Could not read image from path: {image_input}")
            return image
            
        elif isinstance(image_input, bytes):
            # Try to decode as base64 first
            try:
                decoded = base64.b64decode(image_input)
                buf = io.BytesIO(decoded)
            except:
                # If not base64, treat as raw bytes
                buf = io.BytesIO(image_input)
            
            # Convert to numpy array
            nparr = np.frombuffer(buf.getvalue(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Could not decode image from bytes")
            return image
            
        elif isinstance(image_input, np.ndarray):
            # Direct numpy array input
            return image_input.copy()
            
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def visualize_and_save_analysis(self, image_input: Union[str, Path, bytes, np.ndarray], 
                                  output_dir: Union[str, Path] = 'processed_images') -> tuple:
        """
        Process the image, display analysis, and save processed images.
        """
        # Load image from input
        image = self.load_image(image_input)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Store original dimensions
        orig_height, orig_width = image.shape[:2]
        
        # Create copies for visualization
        visualization_results = {}
        
        # Original image
        if len(image.shape) == 3:
            visualization_results['original'] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            visualization_results['original'] = image
            gray = image
        
        visualization_results['grayscale'] = gray
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        visualization_results['enhanced'] = enhanced
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        visualization_results['denoised'] = denoised
        
        # Blur
        blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
        visualization_results['blurred'] = blurred
        
        # Threshold
        threshold = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        visualization_results['threshold'] = threshold
        
        # Find contours for region of interest
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on original image
        contour_image = visualization_results['original'].copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        visualization_results['contours'] = contour_image
        
        # Create heatmap of intensity
        heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        if len(image.shape) == 3:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        visualization_results['heatmap'] = heatmap
        
        # Create signal intensity mask
        signal_mask = np.zeros_like(gray)
        signal_mask[gray > np.mean(gray)] = 255
        visualization_results['signal_mask'] = signal_mask
        
        # Save all processed images
        for name, img in visualization_results.items():
            # Convert RGB to BGR for saving if necessary
            if name in ['original', 'contours', 'heatmap']:
                save_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                save_img = img
            
            output_path = os.path.join(output_dir, f'{name}.png')
            cv2.imwrite(output_path, save_img)
            print(f"Saved {name} image to: {output_path}")
        
        # Display processed images using matplotlib
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.ravel()

        image_names = ['original', 'grayscale', 'enhanced', 'denoised', 'blurred', 'threshold']
        for i, name in enumerate(image_names):
            axs[i].imshow(visualization_results[name], cmap='gray' if len(visualization_results[name].shape) == 2 else None)
            axs[i].set_title(name.capitalize())
            axs[i].axis('off')

        plt.tight_layout()
        plt.show()
        
        # Calculate features
        features = self.extract_features(threshold, gray)
        
        return visualization_results, features

    def analyze_and_save(self, image_input: Union[str, Path, bytes, np.ndarray], 
                        output_dir: Union[str, Path] = 'processed_images') -> list:
        """
        Load, analyze, and save processed images.
        """
        print(f"\nProcessing image...")
        visualization_results, features = self.visualize_and_save_analysis(image_input, output_dir)
        
        # Print feature analysis
        print("\nFeature Analysis:")
        feature_names = [
            "Mean Intensity",
            "Max Intensity",
            "Signal Area",
            "Standard Deviation",
            "Median Intensity",
            "Signal-to-Noise Ratio",
            "Intensity Ratio",
            "Contrast",
            "Homogeneity",
            "Background Intensity",
            "Min Intensity"
        ]
        
        for name, value in zip(feature_names, features):
            print(f"{name}: {value:.2f}")
        
        return features

    def extract_features(self, processed_image, original_gray):
        """Extract features from the processed image."""
        mean_intensity = np.mean(original_gray)
        max_intensity = np.max(original_gray)
        min_intensity = np.min(original_gray)
        std_intensity = np.std(original_gray)
        median_intensity = np.median(original_gray)
        signal_mask = processed_image > 0
        signal_area = np.sum(signal_mask)
        background_mask = ~signal_mask
        background_intensity = np.mean(original_gray[background_mask])
        signal_intensity = np.mean(original_gray[signal_mask])
        signal_to_noise = (signal_intensity - background_intensity) / std_intensity if std_intensity > 0 else 0
        intensity_ratio = max_intensity / (background_intensity + 1e-6)
        glcm = self._calculate_glcm(original_gray)
        contrast = self._calculate_contrast(glcm)
        homogeneity = self._calculate_homogeneity(glcm)
        return [
            mean_intensity,
            max_intensity,
            signal_area,
            std_intensity,
            median_intensity,
            signal_to_noise,
            intensity_ratio,
            contrast,
            homogeneity,
            background_intensity,
            min_intensity
        ]

    def _calculate_glcm(self, image):
        normalized_img = ((image - image.min()) * (255 / (image.max() - image.min()))).astype(np.uint8)
        glcm = np.zeros((256, 256))
        rows, cols = normalized_img.shape
        for i in range(rows-1):
            for j in range(cols-1):
                intensity = normalized_img[i,j]
                neighbor = normalized_img[i,j+1]
                glcm[intensity, neighbor] += 1
        return glcm / glcm.sum()

    def _calculate_contrast(self, glcm):
        contrast = 0
        rows, cols = glcm.shape
        for i in range(rows):
            for j in range(cols):
                contrast += glcm[i,j] * (i-j)**2
        return contrast

    def _calculate_homogeneity(self, glcm):
        homogeneity = 0
        rows, cols = glcm.shape
        for i in range(rows):
            for j in range(cols):
                homogeneity += glcm[i,j] / (1 + abs(i-j))
        return homogeneity
    # Class definition and methods go here...

if __name__ == "__main__":
    # Instantiate the analyzer
    analyzer = FluorescentImmunoassayAnalyzer()
    
    # Specify the file path
    file_path = r"/Users/alexvaisman/Desktop/Screenshot 2025-01-17 at 6.56.30 PM.png"
    
    # Process the image
    try:
        features = analyzer.analyze_and_save(file_path)
        print("\nProcessing complete. Extracted features:")
        for feature in features:
            print(feature)
    except Exception as e:
        print(f"An error occurred: {e}")
