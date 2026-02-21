"""
Local LLM wrapper for scene summary generation.
"""

import logging
import time
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract base class for local LLM implementations."""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model."""
        pass


class MockLocalLLM(BaseLLM):
    """
    Mock LLM implementation for testing and development.
    
    Provides deterministic responses based on prompt analysis
    without requiring actual model loading.
    """
    
    def __init__(self) -> None:
        """Initialize the mock LLM."""
        self.logger = logging.getLogger("wiz.models.local_llm.mock")
        self.logger.info("Initialized Mock LocalLLM for development")
    
    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        """
        Generate mock text based on prompt analysis.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate (ignored in mock)
            
        Returns:
            Generated mock summary text
        """
        # Add small delay to simulate inference time
        time.sleep(0.1)
        
        # Parse key information from prompt
        tone = self._extract_tone_from_prompt(prompt)
        speakers = self._extract_speakers_from_prompt(prompt)
        transcript = self._extract_transcript_from_prompt(prompt)
        
        # Generate mock summary based on extracted information
        summary = self._generate_mock_summary(tone, speakers, transcript)
        
        self.logger.debug(f"Generated mock summary: {summary}")
        return summary
    
    def _extract_tone_from_prompt(self, prompt: str) -> str:
        """Extract tone information from prompt."""
        tone_mapping = {
            "calm": "calm",
            "tense": "tense", 
            "excited": "excited",
            "somber": "somber",
            "neutral": "neutral",
            "confrontational": "confrontational"
        }
        
        for tone_key, tone_value in tone_mapping.items():
            if tone_key in prompt.lower():
                return tone_value
        
        return "neutral"
    
    def _extract_speakers_from_prompt(self, prompt: str) -> list:
        """Extract speaker information from prompt."""
        speakers = []
        
        # Look for speaker patterns
        lines = prompt.split('\n')
        for line in lines:
            if 'Primary Speakers:' in line or 'Speakers:' in line:
                # Extract speaker names/IDs
                speaker_part = line.split(':')[-1].strip()
                if speaker_part and speaker_part != 'None':
                    speakers = speaker_part.split(', ')
                break
        
        return speakers if speakers else ["Speaker"]
    
    def _extract_transcript_from_prompt(self, prompt: str) -> str:
        """Extract transcript excerpt from prompt."""
        lines = prompt.split('\n')
        transcript_started = False
        transcript_lines = []
        
        for line in lines:
            if 'Transcript Excerpt:' in line:
                transcript_started = True
                continue
            elif transcript_started:
                if line.strip() == '' or line.startswith('Write a concise'):
                    break
                transcript_lines.append(line.strip())
        
        return ' '.join(transcript_lines)[:100]  # Limit to first 100 chars
    
    def _generate_mock_summary(self, tone: str, speakers: list, transcript: str) -> str:
        """Generate mock summary based on extracted elements."""
        
        # Template-based summary generation
        if tone == "excited":
            templates = [
                f"{speakers[0] if speakers else 'The speaker'} expresses enthusiasm and high energy. The conversation maintains an upbeat and animated tone throughout the segment.",
                f"An energetic discussion takes place with {speakers[0] if speakers else 'the participant'} showing clear excitement. The mood is positive and engaging."
            ]
        elif tone == "tense":
            templates = [
                f"Tension is evident in the interaction between {', '.join(speakers) if len(speakers) > 1 else speakers[0] if speakers else 'participants'}. The atmosphere suggests underlying stress or conflict.",
                f"A strained conversation unfolds with noticeable hesitation and unease. {speakers[0] if speakers else 'The speaker'} appears nervous or anxious."
            ]
        elif tone == "confrontational":
            templates = [
                f"A heated exchange occurs between {', '.join(speakers) if len(speakers) > 1 else 'the participants'}. The tone is aggressive and argumentative throughout.",
                f"Confrontational dialogue dominates this segment with overlapping speech and raised intensity. Multiple speakers engage in dispute."
            ]
        elif tone == "somber":
            templates = [
                f"{speakers[0] if speakers else 'The speaker'} adopts a serious and subdued tone. The conversation carries weight and gravity.",
                f"A melancholic or contemplative mood pervades the discussion. {speakers[0] if speakers else 'The participant'} speaks with measured, somber delivery."
            ]
        elif tone == "calm":
            templates = [
                f"A composed and steady conversation takes place with {speakers[0] if speakers else 'the speaker'} maintaining a calm demeanor. The interaction is stable and controlled.",
                f"Peaceful dialogue unfolds between {', '.join(speakers) if len(speakers) > 1 else speakers[0] if speakers else 'participants'}. The tone remains even and tranquil."
            ]
        else:  # neutral
            templates = [
                f"A standard conversation takes place with {speakers[0] if speakers else 'the speaker'} maintaining a neutral tone. The interaction follows typical conversational patterns.",
                f"Normal dialogue occurs between {', '.join(speakers) if len(speakers) > 1 else 'the participant'}. The tone is balanced and unremarkable."
            ]
        
        # Select template based on content characteristics
        template_index = hash(transcript + tone) % len(templates)
        
        return templates[template_index]
    
    def get_model_info(self) -> Dict[str, str]:
        """Get mock model information."""
        return {
            "type": "mock",
            "version": "1.0.0",
            "description": "Mock LLM for development and testing",
            "max_tokens": "150",
            "local": "true"
        }


class LlamaCppLLM(BaseLLM):
    """
    LLM implementation using llama.cpp for local inference.
    
    Requires llama-cpp-python to be installed separately.
    """
    
    def __init__(self, 
                 model_path: str,
                 n_ctx: int = 2048,
                 n_threads: int = 4,
                 temperature: float = 0.7) -> None:
        """
        Initialize llama.cpp LLM.
        
        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_threads: Number of CPU threads
            temperature: Sampling temperature
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.temperature = temperature
        self.logger = logging.getLogger("wiz.models.local_llm.llama_cpp")
        
        try:
            from llama_cpp import Llama
            
            self.logger.info(f"Loading llama.cpp model from {model_path}")
            self.model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                verbose=False
            )
            self.logger.info("Successfully loaded llama.cpp model")
            
        except ImportError:
            self.logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            raise ImportError("llama-cpp-python required for LlamaCppLLM")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        """
        Generate text using llama.cpp.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        try:
            # Generate response
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=self.temperature,
                stop=["Human:", "Assistant:", "\n\n"],
                echo=False
            )
            
            generated_text = response['choices'][0]['text'].strip()
            
            self.logger.debug(f"Generated {len(generated_text)} characters")
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return f"Summary generation failed: {str(e)}"
    
    def get_model_info(self) -> Dict[str, str]:
        """Get llama.cpp model information."""
        return {
            "type": "llama_cpp",
            "model_path": self.model_path,
            "context_size": str(self.n_ctx),
            "threads": str(self.n_threads),
            "temperature": str(self.temperature),
            "local": "true"
        }


class MLXLLM(BaseLLM):
    """
    LLM implementation using Apple MLX for local inference.
    
    Optimized for Apple Silicon devices.
    Requires mlx and mlx-lm to be installed separately.
    """
    
    def __init__(self, 
                 model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
                 max_tokens: int = 150) -> None:
        """
        Initialize MLX LLM.
        
        Args:
            model_name: HuggingFace model name or local path
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.logger = logging.getLogger("wiz.models.local_llm.mlx")
        
        try:
            from mlx_lm import load, generate
            
            self.logger.info(f"Loading MLX model: {model_name}")
            self.model, self.tokenizer = load(model_name)
            self.generate_fn = generate
            self.logger.info("Successfully loaded MLX model")
            
        except ImportError:
            self.logger.error("MLX not installed. Install with: pip install mlx mlx-lm")
            raise ImportError("mlx and mlx-lm required for MLXLLM")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        """
        Generate text using MLX.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        try:
            # Format prompt for instruction following
            formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            # Generate response
            response = self.generate_fn(
                self.model,
                self.tokenizer,
                prompt=formatted_prompt,
                max_tokens=max_tokens or self.max_tokens,
                temp=0.7
            )
            
            self.logger.debug(f"Generated response")
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return f"Summary generation failed: {str(e)}"
    
    def get_model_info(self) -> Dict[str, str]:
        """Get MLX model information."""
        return {
            "type": "mlx",
            "model_name": self.model_name,
            "max_tokens": str(self.max_tokens),
            "local": "true",
            "optimized_for": "Apple Silicon"
        }


class LocalLLM:
    """
    Main LocalLLM interface that can use different backends.
    
    Provides a unified interface for local LLM inference with
    swappable backends (mock, llama.cpp, MLX, etc.).
    """
    
    def __init__(self, 
                 backend: str = "mock",
                 model_path: Optional[str] = None,
                 **kwargs) -> None:
        """
        Initialize LocalLLM with specified backend.
        
        Args:
            backend: LLM backend ("mock", "llama_cpp", "mlx")
            model_path: Path to model file (for llama_cpp)
            **kwargs: Additional arguments for specific backends
        """
        self.backend = backend
        self.logger = logging.getLogger("wiz.models.local_llm")
        
        if backend == "mock":
            self.llm = MockLocalLLM()
        elif backend == "llama_cpp":
            if not model_path:
                raise ValueError("model_path required for llama_cpp backend")
            self.llm = LlamaCppLLM(model_path=model_path, **kwargs)
        elif backend == "mlx":
            self.llm = MLXLLM(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        self.logger.info(f"Initialized LocalLLM with {backend} backend")
    
    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        return self.llm.generate(prompt, max_tokens)
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the current model."""
        info = self.llm.get_model_info()
        info["backend"] = self.backend
        return info