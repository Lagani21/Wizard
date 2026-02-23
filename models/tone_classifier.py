"""
Emotional tone classification using structured multimodal features.
"""

import logging
import numpy as np
from typing import Dict, Tuple, List, Optional
from enum import Enum
from abc import ABC, abstractmethod


class ToneLabel(Enum):
    """Enumeration of emotional tone categories."""
    CALM = "calm"
    TENSE = "tense"
    EXCITED = "excited"
    SOMBER = "somber"
    NEUTRAL = "neutral"
    CONFRONTATIONAL = "confrontational"


class BaseToneClassifier(ABC):
    """Abstract base class for tone classifiers."""
    
    @abstractmethod
    def predict(self, features: Dict[str, float]) -> Tuple[ToneLabel, float]:
        """
        Predict emotional tone from features.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Tuple of (predicted_tone, confidence)
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the classifier model."""
        pass


class RuleBasedToneClassifier(BaseToneClassifier):
    """
    Rule-based emotional tone classifier using feature thresholds.
    
    Uses hand-crafted rules to classify emotional tone based on
    multimodal feature combinations.
    """
    
    def __init__(self) -> None:
        """Initialize the rule-based classifier."""
        self.logger = logging.getLogger("wiz.models.tone_classifier.rule_based")
        
        # Define feature thresholds for different tones
        self.thresholds = {
            # Speech rate thresholds
            "high_speech_rate": 3.0,      # words per second
            "low_speech_rate": 1.0,
            
            # Energy thresholds
            "high_energy": 0.1,
            "low_energy": 0.02,
            
            # Motion thresholds
            "high_motion": 2.0,
            "low_motion": 0.5,
            
            # Sentiment thresholds
            "positive_sentiment": 0.1,
            "negative_sentiment": -0.1,
            
            # Speaker interaction thresholds
            "multiple_speakers": 1.5,
            "high_turn_rate": 10.0,        # turns per minute
            
            # Intensity thresholds
            "high_intensity_words": 0.05,
            "high_speaker_overlap": 0.1
        }
    
    def predict(self, features: Dict[str, float]) -> Tuple[ToneLabel, float]:
        """
        Predict emotional tone using rule-based logic.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Tuple of (predicted_tone, confidence)
        """
        # Extract key features with defaults
        speech_rate = features.get("speech_rate_wps", 0.0)
        energy_mean = features.get("energy_mean", 0.0)
        motion_magnitude = features.get("motion_magnitude", 0.0)
        sentiment_polarity = features.get("sentiment_polarity", 0.0)
        speaker_count = features.get("speaker_count", 0.0)
        speaker_turn_rate = features.get("speaker_turn_rate", 0.0)
        intensity_ratio = features.get("intensity_word_ratio", 0.0)
        speaker_overlap = features.get("speaker_overlap_ratio", 0.0)
        hesitation_rate = features.get("hesitation_rate", 0.0)
        exclamation_density = features.get("exclamation_density", 0.0)
        
        # Rule-based classification
        predictions = []
        
        # EXCITED: High speech rate, high energy, high motion, positive sentiment
        if (speech_rate > self.thresholds["high_speech_rate"] and
            energy_mean > self.thresholds["high_energy"] and
            (motion_magnitude > self.thresholds["high_motion"] or 
             exclamation_density > 0.5)):
            excitement_score = self._calculate_excitement_score(features)
            predictions.append((ToneLabel.EXCITED, excitement_score))
        
        # TENSE: High speech rate, moderate energy, high hesitation, multiple speakers
        if ((speech_rate > self.thresholds["high_speech_rate"] or hesitation_rate > 0.3) and
            speaker_count > 1 and
            (speaker_turn_rate > self.thresholds["high_turn_rate"] or 
             speaker_overlap > self.thresholds["high_speaker_overlap"])):
            tension_score = self._calculate_tension_score(features)
            predictions.append((ToneLabel.TENSE, tension_score))
        
        # CONFRONTATIONAL: Multiple speakers, high overlap, negative sentiment, high intensity
        if (speaker_count > 1 and
            speaker_overlap > self.thresholds["high_speaker_overlap"] and
            (sentiment_polarity < self.thresholds["negative_sentiment"] or
             intensity_ratio > self.thresholds["high_intensity_words"])):
            confrontation_score = self._calculate_confrontation_score(features)
            predictions.append((ToneLabel.CONFRONTATIONAL, confrontation_score))
        
        # SOMBER: Low speech rate, low energy, negative sentiment, low motion
        if (speech_rate < self.thresholds["low_speech_rate"] and
            energy_mean < self.thresholds["high_energy"] and
            motion_magnitude < self.thresholds["high_motion"] and
            sentiment_polarity < 0):
            somber_score = self._calculate_somber_score(features)
            predictions.append((ToneLabel.SOMBER, somber_score))
        
        # CALM: Moderate speech rate, stable energy, positive/neutral sentiment, low motion
        if (self.thresholds["low_speech_rate"] <= speech_rate <= self.thresholds["high_speech_rate"] and
            energy_mean > self.thresholds["low_energy"] and
            motion_magnitude < self.thresholds["high_motion"] and
            sentiment_polarity >= 0 and
            hesitation_rate < 0.2):
            calm_score = self._calculate_calm_score(features)
            predictions.append((ToneLabel.CALM, calm_score))
        
        # Select best prediction or default to NEUTRAL
        if predictions:
            # Sort by confidence and return highest
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[0]
        else:
            # Default to NEUTRAL with moderate confidence
            neutral_confidence = self._calculate_neutral_confidence(features)
            return ToneLabel.NEUTRAL, neutral_confidence
    
    def _calculate_excitement_score(self, features: Dict[str, float]) -> float:
        """Calculate confidence score for excitement classification."""
        speech_rate = features.get("speech_rate_wps", 0.0)
        energy_mean = features.get("energy_mean", 0.0)
        motion_magnitude = features.get("motion_magnitude", 0.0)
        sentiment_polarity = features.get("sentiment_polarity", 0.0)
        
        # Normalize features and combine
        speech_factor = min(speech_rate / 5.0, 1.0)  # Cap at 5 wps
        energy_factor = min(energy_mean / 0.2, 1.0)  # Cap at 0.2
        motion_factor = min(motion_magnitude / 3.0, 1.0)  # Cap at 3.0
        sentiment_factor = max(0, sentiment_polarity)  # Only positive sentiment
        
        confidence = (speech_factor + energy_factor + motion_factor + sentiment_factor) / 4.0
        return min(confidence, 0.95)  # Cap confidence
    
    def _calculate_tension_score(self, features: Dict[str, float]) -> float:
        """Calculate confidence score for tension classification."""
        hesitation_rate = features.get("hesitation_rate", 0.0)
        speaker_turn_rate = features.get("speaker_turn_rate", 0.0)
        speaker_overlap = features.get("speaker_overlap_ratio", 0.0)
        energy_variance = features.get("energy_variance", 0.0)
        
        # Combine tension indicators
        hesitation_factor = min(hesitation_rate * 3.0, 1.0)  # Scale hesitation
        turn_factor = min(speaker_turn_rate / 20.0, 1.0)  # Cap at 20 turns/min
        overlap_factor = min(speaker_overlap / 0.2, 1.0)  # Cap at 20% overlap
        variance_factor = min(energy_variance / 0.01, 1.0)  # Energy instability
        
        confidence = (hesitation_factor + turn_factor + overlap_factor + variance_factor) / 4.0
        return min(confidence, 0.9)
    
    def _calculate_confrontation_score(self, features: Dict[str, float]) -> float:
        """Calculate confidence score for confrontation classification."""
        speaker_overlap = features.get("speaker_overlap_ratio", 0.0)
        sentiment_polarity = features.get("sentiment_polarity", 0.0)
        intensity_ratio = features.get("intensity_word_ratio", 0.0)
        exclamation_density = features.get("exclamation_density", 0.0)
        
        # Combine confrontational indicators
        overlap_factor = min(speaker_overlap / 0.15, 1.0)
        negative_sentiment = max(0, -sentiment_polarity)  # Only negative sentiment
        intensity_factor = min(intensity_ratio / 0.1, 1.0)
        exclamation_factor = min(exclamation_density, 1.0)
        
        confidence = (overlap_factor + negative_sentiment + intensity_factor + exclamation_factor) / 4.0
        return min(confidence, 0.9)
    
    def _calculate_somber_score(self, features: Dict[str, float]) -> float:
        """Calculate confidence score for somber classification."""
        speech_rate = features.get("speech_rate_wps", 0.0)
        energy_mean = features.get("energy_mean", 0.0)
        sentiment_polarity = features.get("sentiment_polarity", 0.0)
        motion_magnitude = features.get("motion_magnitude", 0.0)
        
        # Combine somber indicators (inverse of excitement)
        slow_speech_factor = max(0, 1.0 - speech_rate / 2.0)
        low_energy_factor = max(0, 1.0 - energy_mean / 0.1)
        negative_sentiment = max(0, -sentiment_polarity)
        low_motion_factor = max(0, 1.0 - motion_magnitude / 1.0)
        
        confidence = (slow_speech_factor + low_energy_factor + negative_sentiment + low_motion_factor) / 4.0
        return min(confidence, 0.85)
    
    def _calculate_calm_score(self, features: Dict[str, float]) -> float:
        """Calculate confidence score for calm classification."""
        speech_rate = features.get("speech_rate_wps", 0.0)
        energy_variance = features.get("energy_variance", 0.0)
        hesitation_rate = features.get("hesitation_rate", 0.0)
        visual_stability = features.get("visual_stability", 0.0)
        
        # Combine calm indicators (stability and moderate levels)
        moderate_speech = 1.0 - abs(speech_rate - 2.0) / 2.0  # Optimal around 2 wps
        stable_energy = max(0, 1.0 - energy_variance / 0.005)
        fluent_speech = max(0, 1.0 - hesitation_rate / 0.1)
        stable_visual = visual_stability  # Already normalized
        
        confidence = (moderate_speech + stable_energy + fluent_speech + stable_visual) / 4.0
        return min(confidence, 0.8)
    
    def _calculate_neutral_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence score for neutral classification."""
        # Neutral is assigned when no strong patterns are detected
        # Lower confidence indicates uncertainty
        return 0.5
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the rule-based classifier."""
        return {
            "type": "rule_based",
            "version": "1.0",
            "features_used": "multimodal",
            "num_classes": str(len(ToneLabel)),
            "description": "Hand-crafted rules for tone classification"
        }


class MLPToneClassifier(BaseToneClassifier):
    """
    Multi-layer perceptron classifier for emotional tone detection.
    
    Small neural network for learning tone patterns from features.
    Note: This is a placeholder implementation - would need training data.
    """
    
    def __init__(self, 
                 hidden_sizes: List[int] = [64, 32],
                 use_pretrained: bool = False) -> None:
        """
        Initialize the MLP classifier.
        
        Args:
            hidden_sizes: List of hidden layer sizes
            use_pretrained: Whether to load pretrained weights (placeholder)
        """
        self.hidden_sizes = hidden_sizes
        self.use_pretrained = use_pretrained
        self.logger = logging.getLogger("wiz.models.tone_classifier.mlp")
        
        # Placeholder for model weights (would be trained on real data)
        self.is_trained = False
        
        if use_pretrained:
            self.logger.warning("Pretrained MLP not available, falling back to rule-based logic")
    
    def predict(self, features: Dict[str, float]) -> Tuple[ToneLabel, float]:
        """
        Predict emotional tone using MLP.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Tuple of (predicted_tone, confidence)
        """
        if not self.is_trained and not self.use_pretrained:
            self.logger.warning("MLP not trained, using rule-based fallback")
            # Fallback to rule-based classification
            fallback_classifier = RuleBasedToneClassifier()
            return fallback_classifier.predict(features)
        
        # Placeholder MLP prediction logic
        # In real implementation, would use trained neural network
        feature_vector = self._features_to_vector(features)
        
        # Mock prediction (would be actual forward pass)
        mock_logits = self._mock_forward_pass(feature_vector)
        
        # Convert to probabilities
        probabilities = self._softmax(mock_logits)
        
        # Get prediction
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        # Map class index to ToneLabel
        tone_labels = list(ToneLabel)
        predicted_tone = tone_labels[predicted_class]
        
        return predicted_tone, float(confidence)
    
    def _features_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to vector."""
        # Expected feature order (would be consistent with training)
        feature_names = [
            "speech_rate_wps", "energy_mean", "motion_magnitude", "sentiment_polarity",
            "speaker_count", "speaker_turn_rate", "intensity_word_ratio",
            "speaker_overlap_ratio", "hesitation_rate", "exclamation_density",
            "visual_stability", "energy_variance"
        ]
        
        # Extract features in consistent order
        feature_vector = []
        for name in feature_names:
            value = features.get(name, 0.0)
            feature_vector.append(value)
        
        return np.array(feature_vector, dtype=np.float32)
    
    def _mock_forward_pass(self, features: np.ndarray) -> np.ndarray:
        """Mock forward pass through MLP (placeholder)."""
        # This would be replaced with actual trained network
        
        # Simple heuristic-based mock predictions
        logits = np.zeros(len(ToneLabel))
        
        # Extract key features
        speech_rate = features[0] if len(features) > 0 else 0
        energy_mean = features[1] if len(features) > 1 else 0
        motion = features[2] if len(features) > 2 else 0
        sentiment = features[3] if len(features) > 3 else 0
        
        # Map enum members to their integer list positions (matches predict() reverse mapping)
        _idx = {label: i for i, label in enumerate(ToneLabel)}

        # Mock classification logic
        if speech_rate > 3.0 and energy_mean > 0.1:
            logits[_idx[ToneLabel.EXCITED]] = 2.0
        elif speech_rate < 1.0 and energy_mean < 0.05:
            logits[_idx[ToneLabel.SOMBER]] = 1.5
        elif motion > 2.0:
            logits[_idx[ToneLabel.TENSE]] = 1.8
        elif sentiment < -0.1:
            logits[_idx[ToneLabel.CONFRONTATIONAL]] = 1.3
        elif 1.0 <= speech_rate <= 3.0:
            logits[_idx[ToneLabel.CALM]] = 1.0
        else:
            logits[_idx[ToneLabel.NEUTRAL]] = 0.8
        
        return logits
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        return exp_logits / np.sum(exp_logits)
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the MLP classifier."""
        return {
            "type": "mlp",
            "version": "1.0",
            "hidden_sizes": str(self.hidden_sizes),
            "features_used": "multimodal",
            "num_classes": str(len(ToneLabel)),
            "trained": str(self.is_trained),
            "description": "Multi-layer perceptron for tone classification"
        }


class ToneClassifier:
    """
    Main tone classifier that can use different backend classifiers.
    
    Provides a unified interface for emotional tone classification
    using either rule-based or MLP approaches.
    """
    
    def __init__(self, 
                 classifier_type: str = "rule_based",
                 **kwargs) -> None:
        """
        Initialize the tone classifier.
        
        Args:
            classifier_type: Type of classifier ("rule_based" or "mlp")
            **kwargs: Additional arguments for the specific classifier
        """
        self.classifier_type = classifier_type
        self.logger = logging.getLogger("wiz.models.tone_classifier")
        
        # Initialize the appropriate classifier
        if classifier_type == "rule_based":
            self.classifier = RuleBasedToneClassifier()
        elif classifier_type == "mlp":
            self.classifier = MLPToneClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        self.logger.info(f"Initialized {classifier_type} tone classifier")
    
    def predict(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Predict emotional tone from features.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Tuple of (tone_label_string, confidence)
        """
        tone_enum, confidence = self.classifier.predict(features)
        return tone_enum.value, confidence
    
    def get_supported_tones(self) -> List[str]:
        """Get list of supported tone labels."""
        return [tone.value for tone in ToneLabel]
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the classifier."""
        info = self.classifier.get_model_info()
        info["classifier_type"] = self.classifier_type
        return info