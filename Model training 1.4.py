import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import pandas as pd
import sqlite3
import h5py
import lmdb
from pathlib import Path
import psycopg2
from sqlalchemy import create_engine
import threading
from queue import Queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import pymongo
from tqdm import tqdm
import os
import time
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class DatabaseConfig:
    def __init__(self, db_type, connection_params):
        self.db_type = db_type.lower()
        self.connection_params = connection_params
        self.supported_dbs = ['sqlite', 'postgres', 'mongodb', 'hdf5', 'lmdb']
        
        if self.db_type not in self.supported_dbs:
            raise ValueError(f"Unsupported database type. Supported types: {self.supported_dbs}")

class ImageDataLoader:
    def __init__(self, db_config, batch_size=32, num_workers=4):
        self.db_config = db_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.queue = Queue(maxsize=100)
        self.setup_database_connection()

    def setup_database_connection(self):
        if self.db_config.db_type == 'sqlite':
            self.conn = sqlite3.connect(self.db_config.connection_params['db_path'])
        elif self.db_config.db_type == 'postgres':
            self.engine = create_engine(
                f"postgresql://{self.db_config.connection_params['user']}:"
                f"{self.db_config.connection_params['password']}@"
                f"{self.db_config.connection_params['host']}:"
                f"{self.db_config.connection_params['port']}/"
                f"{self.db_config.connection_params['database']}"
            )
        elif self.db_config.db_type == 'mongodb':
            self.client = pymongo.MongoClient(self.db_config.connection_params['connection_string'])
            self.db = self.client[self.db_config.connection_params['database']]
        elif self.db_config.db_type == 'hdf5':
            self.h5_file = h5py.File(self.db_config.connection_params['file_path'], 'a')
        elif self.db_config.db_type == 'lmdb':
            self.env = lmdb.open(self.db_config.connection_params['path'],
                               map_size=1099511627776)

class EnsembleClassifier:
    def __init__(self, voting='soft', n_jobs=-1):
        self.rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            n_jobs=n_jobs,
            random_state=42
        )
        
        self.svm = SVC(
            kernel='rbf',
            probability=True,
            C=1.0,
            random_state=42
        )
        
        self.gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.ensemble = VotingClassifier(
            estimators=[
                ('rf', self.rf),
                ('svm', self.svm),
                ('gb', self.gb)
            ],
            voting=voting,
            n_jobs=n_jobs
        )
        
        self.individual_predictions = {}
        
    def fit(self, X, y):
        self.ensemble.fit(X, y)
        return self
        
    def predict(self, X):
        self.individual_predictions = {
            'RandomForest': self.rf.predict(X),
            'SVM': self.svm.predict(X),
            'GradientBoosting': self.gb.predict(X)
        }
        return self.ensemble.predict(X)
    
    def predict_proba(self, X):
        return self.ensemble.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        results = {}
        ensemble_pred = self.predict(X_test)
        
        results['ensemble'] = {
            'accuracy': accuracy_score(y_test, ensemble_pred),
            'classification_report': classification_report(y_test, ensemble_pred),
            'confusion_matrix': confusion_matrix(y_test, ensemble_pred)
        }
        
        for model_name, predictions in self.individual_predictions.items():
            results[model_name] = {
                'accuracy': accuracy_score(y_test, predictions),
                'classification_report': classification_report(y_test, predictions),
                'confusion_matrix': confusion_matrix(y_test, predictions)
            }
            
        return results

class PyTorchModel(nn.Module):
    def __init__(self, num_classes=2):
        super(PyTorchModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_features = self.resnet.fc.in_features
        
        # Feature extraction layers
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)
    
    def extract_features(self, x):
        features = self.features(x)
        return features.view(features.size(0), -1)

class EnhancedFluorescentImmunoassayAI:
    def __init__(self, framework='pytorch', db_config=None, use_ensemble=True):
        self.framework = framework
        self.db_config = db_config
        self.use_ensemble = use_ensemble
        
        # Initialize confidence thresholds
        self.confidence_levels = {
            'very_high': 0.90,
            'high': 0.80,
            'moderate': 0.65,
            'low': 0.50
        }
        
        # Initialize models
        if framework == 'pytorch':
            self.init_pytorch_model()
        else:
            self.init_tensorflow_model()
            
        # Initialize ensemble if enabled
        if use_ensemble:
            self.ensemble_classifier = EnsembleClassifier()
        
        # Setup data transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Initialize data loader if database config provided
        if db_config:
            self.data_loader = ImageDataLoader(db_config)

    def init_pytorch_model(self):
        self.model = PyTorchModel()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def init_tensorflow_model(self):
        base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        predictions = Dense(2, activation='softmax')(x)
        self.model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def preprocess_image(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        return image

    def extract_features(self, images):
        features = []
        self.model.eval()
        
        with torch.no_grad():
            for image in tqdm(images, desc="Extracting features"):
                if isinstance(image, str):
                    image = self.preprocess_image(image)
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                features.append(self.model.extract_features(image_tensor).cpu().numpy())
                
        return np.vstack(features)

    def train_model(self, image_paths, labels, epochs=10, batch_size=32):
        X_train, X_val, y_train, y_val = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42
        )
        
        # Train deep learning model
        if self.framework == 'pytorch':
            train_dataset = self.create_dataset(X_train, y_train)
            val_dataset = self.create_dataset(X_val, y_val)
            dl_results = self._train_pytorch(train_dataset, val_dataset, epochs, batch_size)
        else:
            dl_results = self._train_tensorflow(X_train, y_train, X_val, y_val, epochs, batch_size)
        
        # Train ensemble if enabled
        if self.use_ensemble:
            features_train = self.extract_features([self.preprocess_image(img) for img in X_train])
            features_val = self.extract_features([self.preprocess_image(img) for img in X_val])
            ensemble_results = self.train_ensemble(features_train, y_train, features_val, y_val)
            return {'deep_learning': dl_results, 'ensemble': ensemble_results}
        
        return {'deep_learning': dl_results}

    def train_ensemble(self, X_train, y_train, X_val, y_val):
        self.ensemble_classifier.fit(X_train, y_train)
        return self.ensemble_classifier.evaluate(X_val, y_val)

    def _train_pytorch(self, train_dataset, val_dataset, epochs, batch_size):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            # Record metrics
            history['train_loss'].append(train_loss/len(train_loader))
            history['val_loss'].append(val_loss/len(val_loader))
            history['val_accuracy'].append(100 * correct / total)
            
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Training Loss: {history["train_loss"][-1]:.4f}')
            print(f'Validation Loss: {history["val_loss"][-1]:.4f}')
            print(f'Validation Accuracy: {history["val_accuracy"][-1]:.2f}%\n')
        
        return history

    def predict(self, image_path, return_confidence=True):
        image = self.preprocess_image(image_path)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Deep learning prediction
        self.model.eval()
        with torch.no_grad():
            dl_outputs = self.model(image_tensor)
            dl_probs = torch.softmax(dl_outputs, dim=1)
            dl_pred = torch.argmax(dl_probs, dim=1).item()
            dl_confidence = dl_probs[0][dl_pred].item()
        
        # Ensemble prediction if enabled
        if self.use_ensemble:
            features = self.extract_features([image])
            ensemble_pred = self.ensemble_classifier.predict(features)[0]
            ensemble_probs = self.ensemble_classifier.predict_proba(features)[0]
            ensemble_confidence = ensemble_probs[ensemble_pred]
            
            # Combine predictions
            final_pred = dl_pred if dl_confidence > ensemble_confidence else ensemble_pred
            final_confidence = max(dl_confidence, ensemble_confidence)
        else:
            final_pred = dl_pred
            final_confidence = dl_confidence
        
        if return_confidence:
            confidence_level = next(
                level for level, threshold in self.confidence_levels.items()
                if final_confidence >= threshold
            )
            return final_pred, final_confidence, confidence_level
        
        return final_pred

    def visualize_results(self, history):
        plt.figure(figsize=(12, 4))
        
        # Plot training history
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt
def visualize_results(self, history):
        plt.figure(figsize=(12, 4))
        
        # Plot training history
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def visualize_confusion_matrix(self, y_true, y_pred, class_names=None):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names if class_names else 'auto',
                   yticklabels=class_names if class_names else 'auto')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    def create_dataset(self, image_paths, labels):
        """Create a PyTorch dataset from image paths and labels"""
        images = [self.preprocess_image(path) for path in image_paths]
        return ImageDataset(images, labels, self.transform)

    def save_model(self, path):
        """Save the trained model and ensemble classifier"""
        model_state = {
            'deep_learning_state': self.model.state_dict(),
            'ensemble_classifier': self.ensemble_classifier if self.use_ensemble else None,
            'transform': self.transform,
            'confidence_levels': self.confidence_levels
        }
        torch.save(model_state, path)

    def load_model(self, path):
        """Load a trained model and ensemble classifier"""
        model_state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(model_state['deep_learning_state'])
        if model_state['ensemble_classifier']:
            self.ensemble_classifier = model_state['ensemble_classifier']
            self.use_ensemble = True
        self.transform = model_state['transform']
        self.confidence_levels = model_state['confidence_levels']

    def batch_predict(self, image_paths, batch_size=32):
        """Make predictions on a batch of images"""
        predictions = []
        confidences = []
        confidence_levels = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = [self.preprocess_image(path) for path in batch_paths]
            batch_tensors = torch.stack([self.transform(img) for img in batch_images]).to(self.device)
            
            with torch.no_grad():
                # Deep learning predictions
                outputs = self.model(batch_tensors)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                confs = torch.max(probs, dim=1)[0]
                
                if self.use_ensemble:
                    # Ensemble predictions
                    features = self.extract_features(batch_images)
                    ensemble_preds = self.ensemble_classifier.predict(features)
                    ensemble_probs = self.ensemble_classifier.predict_proba(features)
                    ensemble_confs = np.max(ensemble_probs, axis=1)
                    
                    # Combine predictions based on confidence
                    for dl_pred, dl_conf, ens_pred, ens_conf in zip(
                        preds, confs, ensemble_preds, ensemble_confs):
                        if dl_conf > ens_conf:
                            predictions.append(dl_pred.item())
                            confidences.append(dl_conf.item())
                        else:
                            predictions.append(ens_pred)
                            confidences.append(ens_conf)
                else:
                    predictions.extend(preds.cpu().numpy())
                    confidences.extend(confs.cpu().numpy())
                
                # Determine confidence levels
                for conf in confidences[-len(batch_paths):]:
                    level = next(
                        level for level, threshold in self.confidence_levels.items()
                        if conf >= threshold
                    )
                    confidence_levels.append(level)
        
        return predictions, confidences, confidence_levels

class ImageDataset(Dataset):
    """PyTorch Dataset for immunoassay images"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def example_usage():
    """Example usage of the Enhanced Fluorescent Immunoassay AI system"""
    
    # Initialize the system
    ai_system = EnhancedFluorescentImmunoassayAI(
        framework='pytorch',
        use_ensemble=True
    )
    
    # Sample data
    image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']
    labels = [0, 1]  # Binary classification example
    
    # Train the model
    training_results = ai_system.train_model(
        image_paths=image_paths,
        labels=labels,
        epochs=10,
        batch_size=32
    )
    
    # Visualize training results
    ai_system.visualize_results(training_results['deep_learning'])
    
    # Make predictions
    for image_path in image_paths:
        prediction, confidence, confidence_level = ai_system.predict(
            image_path,
            return_confidence=True
        )
        print(f"Image: {image_path}")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Confidence Level: {confidence_level}")
        print()
    
    # Save the model
    ai_system.save_model('immunoassay_model.pth')

if __name__ == "__main__":
    example_usage()
