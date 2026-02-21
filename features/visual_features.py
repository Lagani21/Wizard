"""
Visual feature extraction from VideoMAE embeddings for emotional tone detection.
"""

import logging
import numpy as np
from typing import List, Dict, Optional
from ..core.context import VisualEmbedding


class VisualFeatureExtractor:
    """
    Extracts visual-based features from VideoMAE embeddings for tone detection.
    
    Analyzes motion patterns, scene intensity, and visual changes
    that correlate with emotional tone.
    """
    
    def __init__(self) -> None:
        """Initialize the visual feature extractor."""
        self.logger = logging.getLogger("wiz.features.visual")
    
    def extract_features(self, 
                        visual_embeddings: List[VisualEmbedding],
                        time_window_start: float,
                        time_window_end: float) -> Dict[str, float]:
        """
        Extract visual features for a given time window.
        
        Args:
            visual_embeddings: List of visual embeddings with timestamps
            time_window_start: Start time of analysis window
            time_window_end: End time of analysis window
            
        Returns:
            Dictionary of extracted visual features
        """
        # Find embeddings that overlap with the time window
        window_embeddings = self._get_embeddings_in_window(
            visual_embeddings, time_window_start, time_window_end
        )
        
        if not window_embeddings:
            return self._get_empty_features()
        
        # Extract various visual features
        features = {}
        
        # Motion and activity features
        features.update(self._extract_motion_features(window_embeddings))
        
        # Scene intensity features  
        features.update(self._extract_intensity_features(window_embeddings))
        
        # Temporal consistency features
        features.update(self._extract_consistency_features(window_embeddings))
        
        # Embedding distribution features
        features.update(self._extract_distribution_features(window_embeddings))
        
        self.logger.debug(f"Extracted {len(features)} visual features for window {time_window_start:.1f}-{time_window_end:.1f}s")
        
        return features
    
    def _get_embeddings_in_window(self, 
                                 embeddings: List[VisualEmbedding],
                                 start_time: float,
                                 end_time: float) -> List[VisualEmbedding]:
        """Get embeddings that overlap with the time window."""
        window_embeddings = []
        
        for embedding in embeddings:
            # Check for overlap
            if (embedding.start_time < end_time and embedding.end_time > start_time):
                window_embeddings.append(embedding)
        
        # Sort by start time
        window_embeddings.sort(key=lambda x: x.start_time)
        
        return window_embeddings
    
    def _extract_motion_features(self, embeddings: List[VisualEmbedding]) -> Dict[str, float]:
        """Extract motion and activity-related features."""
        if len(embeddings) < 2:
            return {
                "motion_magnitude": 0.0,
                "motion_variance": 0.0,
                "motion_acceleration": 0.0
            }
        
        # Calculate motion as embedding differences between consecutive shots
        motion_magnitudes = []
        
        for i in range(len(embeddings) - 1):
            emb1 = embeddings[i].embedding
            emb2 = embeddings[i + 1].embedding
            
            # Ensure embeddings have same size
            if emb1.shape != emb2.shape:
                min_size = min(len(emb1), len(emb2))
                emb1 = emb1[:min_size]
                emb2 = emb2[:min_size]
            
            # Calculate L2 distance between embeddings
            motion = np.linalg.norm(emb2 - emb1)
            motion_magnitudes.append(motion)
        
        motion_magnitudes = np.array(motion_magnitudes)
        
        # Motion statistics
        motion_magnitude = np.mean(motion_magnitudes)
        motion_variance = np.var(motion_magnitudes)
        
        # Motion acceleration (second derivative)
        if len(motion_magnitudes) > 1:
            motion_diff = np.diff(motion_magnitudes)
            motion_acceleration = np.mean(np.abs(motion_diff))
        else:
            motion_acceleration = 0.0
        
        return {
            "motion_magnitude": float(motion_magnitude),
            "motion_variance": float(motion_variance), 
            "motion_acceleration": float(motion_acceleration)
        }
    
    def _extract_intensity_features(self, embeddings: List[VisualEmbedding]) -> Dict[str, float]:
        """Extract scene intensity and energy features."""
        if not embeddings:
            return {
                "scene_intensity_mean": 0.0,
                "scene_intensity_variance": 0.0,
                "scene_energy_peaks": 0.0
            }
        
        # Calculate intensity as embedding magnitude
        intensities = []
        
        for embedding in embeddings:
            # Use L2 norm of embedding as intensity proxy
            intensity = np.linalg.norm(embedding.embedding)
            intensities.append(intensity)
        
        intensities = np.array(intensities)
        
        # Intensity statistics
        intensity_mean = np.mean(intensities)
        intensity_variance = np.var(intensities)
        
        # Count intensity peaks (above mean + std)
        if len(intensities) > 2:
            threshold = intensity_mean + np.std(intensities)
            peaks = np.sum(intensities > threshold)
            peak_ratio = peaks / len(intensities)
        else:
            peak_ratio = 0.0
        
        return {
            "scene_intensity_mean": float(intensity_mean),
            "scene_intensity_variance": float(intensity_variance),
            "scene_energy_peaks": float(peak_ratio)
        }
    
    def _extract_consistency_features(self, embeddings: List[VisualEmbedding]) -> Dict[str, float]:
        """Extract temporal consistency and stability features."""
        if len(embeddings) < 3:
            return {
                "visual_stability": 0.0,
                "shot_similarity": 0.0,
                "temporal_smoothness": 0.0
            }
        
        # Calculate shot-to-shot similarities
        similarities = []
        
        for i in range(len(embeddings) - 1):
            emb1 = embeddings[i].embedding
            emb2 = embeddings[i + 1].embedding
            
            # Ensure embeddings have same size
            if emb1.shape != emb2.shape:
                min_size = min(len(emb1), len(emb2))
                emb1 = emb1[:min_size]
                emb2 = emb2[:min_size]
            
            # Calculate cosine similarity
            if np.linalg.norm(emb1) > 0 and np.linalg.norm(emb2) > 0:
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                similarities.append(similarity)
        
        if not similarities:
            return {
                "visual_stability": 0.0,
                "shot_similarity": 0.0,
                "temporal_smoothness": 0.0
            }
        
        similarities = np.array(similarities)
        
        # Visual stability (high similarity = stable scene)
        visual_stability = np.mean(similarities)
        
        # Average shot similarity
        shot_similarity = np.mean(similarities)
        
        # Temporal smoothness (low variance in similarities = smooth)
        temporal_smoothness = 1.0 / (1.0 + np.var(similarities))  # Higher = smoother
        
        return {
            "visual_stability": float(visual_stability),
            "shot_similarity": float(shot_similarity),
            "temporal_smoothness": float(temporal_smoothness)
        }
    
    def _extract_distribution_features(self, embeddings: List[VisualEmbedding]) -> Dict[str, float]:
        """Extract features from embedding distribution characteristics."""
        if not embeddings:
            return {
                "embedding_dimensionality": 0.0,
                "embedding_sparsity": 0.0,
                "embedding_entropy": 0.0
            }
        
        # Stack all embeddings
        all_embeddings = []
        for embedding in embeddings:
            all_embeddings.append(embedding.embedding)
        
        if not all_embeddings:
            return {
                "embedding_dimensionality": 0.0,
                "embedding_sparsity": 0.0,
                "embedding_entropy": 0.0
            }
        
        # Find common dimensionality
        min_dim = min(len(emb) for emb in all_embeddings)
        
        # Truncate to common size and stack
        truncated_embeddings = [emb[:min_dim] for emb in all_embeddings]
        stacked_embeddings = np.stack(truncated_embeddings)
        
        # Effective dimensionality (based on variance)
        embedding_vars = np.var(stacked_embeddings, axis=0)
        variance_threshold = 0.01 * np.max(embedding_vars) if np.max(embedding_vars) > 0 else 0
        effective_dims = np.sum(embedding_vars > variance_threshold)
        dimensionality_ratio = effective_dims / len(embedding_vars) if len(embedding_vars) > 0 else 0
        
        # Sparsity (proportion of near-zero values)
        flattened = stacked_embeddings.flatten()
        sparsity = np.sum(np.abs(flattened) < 0.01) / len(flattened) if len(flattened) > 0 else 0
        
        # Entropy-like measure (distribution uniformity)
        if len(flattened) > 0:
            # Discretize values for entropy calculation
            hist, _ = np.histogram(flattened, bins=50)
            hist = hist + 1e-12  # Avoid log(0)
            probs = hist / np.sum(hist)
            entropy = -np.sum(probs * np.log(probs))
            entropy_normalized = entropy / np.log(len(probs))  # Normalize to [0, 1]
        else:
            entropy_normalized = 0.0
        
        return {
            "embedding_dimensionality": float(dimensionality_ratio),
            "embedding_sparsity": float(sparsity),
            "embedding_entropy": float(entropy_normalized)
        }
    
    def _get_empty_features(self) -> Dict[str, float]:
        """Return empty feature set when no visual data is available."""
        return {
            "motion_magnitude": 0.0,
            "motion_variance": 0.0,
            "motion_acceleration": 0.0,
            "scene_intensity_mean": 0.0,
            "scene_intensity_variance": 0.0,
            "scene_energy_peaks": 0.0,
            "visual_stability": 0.0,
            "shot_similarity": 0.0,
            "temporal_smoothness": 0.0,
            "embedding_dimensionality": 0.0,
            "embedding_sparsity": 0.0,
            "embedding_entropy": 0.0
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names extracted by this class."""
        return list(self._get_empty_features().keys())
    
    def create_placeholder_embeddings(self, 
                                    video_duration: float,
                                    shot_duration: float = 2.0,
                                    embedding_dim: int = 768) -> List[VisualEmbedding]:
        """
        Create placeholder visual embeddings for testing.
        
        Args:
            video_duration: Total video duration in seconds
            shot_duration: Duration of each shot in seconds
            embedding_dim: Dimensionality of embeddings
            
        Returns:
            List of placeholder VisualEmbedding objects
        """
        embeddings = []
        
        current_time = 0.0
        shot_id = 0
        
        while current_time < video_duration:
            end_time = min(current_time + shot_duration, video_duration)
            
            # Generate random embedding (placeholder)
            embedding = np.random.normal(0, 1, embedding_dim).astype(np.float32)
            
            visual_embedding = VisualEmbedding(
                start_time=current_time,
                end_time=end_time,
                embedding=embedding,
                shot_id=f"shot_{shot_id:03d}"
            )
            
            embeddings.append(visual_embedding)
            
            current_time = end_time
            shot_id += 1
        
        self.logger.info(f"Created {len(embeddings)} placeholder visual embeddings for {video_duration:.1f}s video")
        
        return embeddings