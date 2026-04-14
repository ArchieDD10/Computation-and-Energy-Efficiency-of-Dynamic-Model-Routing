"""
Temperature Scaling for Model Calibration
Author: Archie Deguzman
Purpose: Improve confidence calibration for better routing decisions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import brier_score_loss
from typing import Dict, List, Tuple
import json
import os

class TemperatureScaling:
    """
    Implements temperature scaling for neural network calibration
    """

    def __init__(self, model_pipeline):
        self.model_pipeline = model_pipeline
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Initialize temperature
        self.device = next(model_pipeline.model.parameters()).device if hasattr(model_pipeline.model, 'parameters') else 'cpu'

    def forward_with_temperature(self, logits):
        """Apply temperature scaling to logits"""
        return torch.softmax(logits / self.temperature, dim=-1)

    def get_logits_and_labels(self, texts: List[str], true_labels: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract logits and labels from model"""
        logits_list = []

        for text in texts:
            # Get raw outputs from the pipeline
            with torch.no_grad():
                # For HuggingFace pipelines, we need to access the model directly
                inputs = self.model_pipeline.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model_pipeline.model(**inputs)
                logits = outputs.logits[0]  # Get first (and only) sequence
                logits_list.append(logits.cpu())

        logits_tensor = torch.stack(logits_list)
        labels_tensor = torch.tensor(true_labels, dtype=torch.long)

        return logits_tensor, labels_tensor

    def calibrate(self, validation_texts: List[str], validation_labels: List[int],
                  max_iter: int = 100, lr: float = 0.001) -> Dict:
        """
        Learn optimal temperature on validation set
        """
        print(f"[INFO] Calibrating temperature for model...")

        # Get logits and labels
        logits, labels = self.get_logits_and_labels(validation_texts, validation_labels)

        # Setup optimizer
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_probs = self.forward_with_temperature(logits)
            loss = nn.CrossEntropyLoss()(scaled_probs, labels)
            loss.backward()
            return loss

        # Optimize temperature
        initial_temp = self.temperature.item()
        optimizer.step(eval_loss)
        final_temp = self.temperature.item()

        # Evaluate calibration metrics
        with torch.no_grad():
            original_probs = torch.softmax(logits, dim=-1)
            scaled_probs = self.forward_with_temperature(logits)

            # Calculate calibration metrics
            original_accuracy = (original_probs.argmax(dim=-1) == labels).float().mean().item()
            scaled_accuracy = (scaled_probs.argmax(dim=-1) == labels).float().mean().item()

            # Calculate confidence scores (max probability)
            original_confidence = original_probs.max(dim=-1)[0].mean().item()
            scaled_confidence = scaled_probs.max(dim=-1)[0].mean().item()

            # Brier score (lower is better)
            original_brier = brier_score_loss(labels.numpy(), original_probs[:,1].numpy()) if logits.shape[1] == 2 else None
            scaled_brier = brier_score_loss(labels.numpy(), scaled_probs[:,1].numpy()) if logits.shape[1] == 2 else None

        results = {
            'initial_temperature': initial_temp,
            'final_temperature': final_temp,
            'original_accuracy': original_accuracy,
            'scaled_accuracy': scaled_accuracy,
            'original_avg_confidence': original_confidence,
            'scaled_avg_confidence': scaled_confidence,
            'original_brier_score': original_brier,
            'scaled_brier_score': scaled_brier,
            'calibration_improvement': original_brier - scaled_brier if original_brier else None
        }

        print(f"[DONE] Temperature calibration complete:")
        print(f"  Temperature: {initial_temp:.3f} -> {final_temp:.3f}")
        print(f"  Accuracy: {original_accuracy:.4f} -> {scaled_accuracy:.4f}")
        print(f"  Avg Confidence: {original_confidence:.4f} -> {scaled_confidence:.4f}")
        if original_brier:
            print(f"  Brier Score: {original_brier:.4f} -> {scaled_brier:.4f}")

        return results

    def predict_with_temperature(self, text: str) -> Dict:
        """Make prediction with temperature scaling"""
        # Get raw logits
        inputs = self.model_pipeline.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model_pipeline.model(**inputs)
            logits = outputs.logits[0]

            # Original prediction
            original_probs = torch.softmax(logits, dim=-1)

            # Temperature-scaled prediction
            scaled_probs = torch.softmax(logits / self.temperature, dim=-1)

            return {
                'original_probs': original_probs.cpu().numpy(),
                'scaled_probs': scaled_probs.cpu().numpy(),
                'original_prediction': int(original_probs.argmax()),
                'scaled_prediction': int(scaled_probs.argmax()),
                'original_confidence': float(original_probs.max()),
                'scaled_confidence': float(scaled_probs.max()),
                'temperature': float(self.temperature.item())
            }


class CalibratedRoutingSystem:
    """
    Enhanced routing system with temperature-calibrated models
    """

    def __init__(self, small_model, medium_model, large_model):
        self.small_model = small_model
        self.medium_model = medium_model
        self.large_model = large_model

        # Temperature scalers for each model
        self.small_scaler = None
        self.medium_scaler = None
        self.large_scaler = None

        # Thresholds (may need adjustment after calibration)
        self.tau_small = 0.90
        self.tau_medium = 0.95

    def calibrate_all_models(self, validation_texts: List[str], validation_labels: List[int]):
        """Calibrate all three models"""
        results = {}

        print("=" * 60)
        print("TEMPERATURE CALIBRATION FOR ALL MODELS")
        print("=" * 60)

        # Calibrate small model
        print("\n[1/3] Calibrating Small Model (DistilBERT)...")
        self.small_scaler = TemperatureScaling(self.small_model)
        results['small'] = self.small_scaler.calibrate(validation_texts, validation_labels)

        # Calibrate medium model
        print("\n[2/3] Calibrating Medium Model (BERT-base)...")
        self.medium_scaler = TemperatureScaling(self.medium_model)
        results['medium'] = self.medium_scaler.calibrate(validation_texts, validation_labels)

        # Calibrate large model
        print("\n[3/3] Calibrating Large Model (RoBERTa-large)...")
        self.large_scaler = TemperatureScaling(self.large_model)
        results['large'] = self.large_scaler.calibrate(validation_texts, validation_labels)

        print("\n" + "=" * 60)
        print("[DONE] ALL MODELS CALIBRATED")
        print("=" * 60)

        return results

    def predict_with_routing_calibrated(self, text: str) -> Dict:
        """Make prediction using calibrated routing"""

        # Try small model first (with calibration)
        if self.small_scaler:
            small_result = self.small_scaler.predict_with_temperature(text)
            small_confidence = small_result['scaled_confidence']
        else:
            small_result = self.small_model(text)
            small_confidence = max(small_result[0]['score'], 1 - small_result[0]['score'])

        if small_confidence >= self.tau_small:
            return {
                'prediction': small_result['scaled_prediction'] if self.small_scaler else small_result[0]['label'],
                'confidence': small_confidence,
                'model_used': 'small',
                'calibrated': bool(self.small_scaler)
            }

        # Try medium model (with calibration)
        if self.medium_scaler:
            medium_result = self.medium_scaler.predict_with_temperature(text)
            medium_confidence = medium_result['scaled_confidence']
        else:
            medium_result = self.medium_model(text)
            medium_confidence = max(medium_result[0]['score'], 1 - medium_result[0]['score'])

        if medium_confidence >= self.tau_medium:
            return {
                'prediction': medium_result['scaled_prediction'] if self.medium_scaler else medium_result[0]['label'],
                'confidence': medium_confidence,
                'model_used': 'medium',
                'calibrated': bool(self.medium_scaler)
            }

        # Use large model (with calibration)
        if self.large_scaler:
            large_result = self.large_scaler.predict_with_temperature(text)
            return {
                'prediction': large_result['scaled_prediction'],
                'confidence': large_result['scaled_confidence'],
                'model_used': 'large',
                'calibrated': True
            }
        else:
            large_result = self.large_model(text)
            return {
                'prediction': large_result[0]['label'],
                'confidence': max(large_result[0]['score'], 1 - large_result[0]['score']),
                'model_used': 'large',
                'calibrated': False
            }


def evaluate_calibration_benefits(original_routing_results: str, calibrated_routing_results: str) -> Dict:
    """Compare performance before and after calibration"""

    # This would load and compare the routing results
    # Implementation would compare accuracy, routing decisions, etc.

    return {
        'accuracy_improvement': 0.02,  # Example improvement
        'routing_efficiency': 0.05,   # Example efficiency gain
        'calibration_quality': 0.15   # Example calibration improvement
    }


if __name__ == "__main__":
    print("Temperature Scaling Module for Model Calibration")
    print("Usage: Import and use CalibratedRoutingSystem with your models")