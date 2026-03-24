"""
Chutes Inference Client for SN85 Scene Classification

Provides remote inference via Chutes API as the primary inference provider,
with automatic fallback to local PyTorch inference when Chutes is unavailable.

Environment variables:
    USE_CHUTES: Set to "true" to enable Chutes inference (default: false for backward compat)
    CHUTES_API_KEY: Your Chutes API key
    CHUTES_SCENE_CHUTE_ID: Chute ID for scene classifier (default: scene-classifier-v1)
    CHUTES_TIMEOUT: Request timeout in seconds (default: 30)
    CHUTES_MAX_RETRIES: Number of retries on failure (default: 2)
"""

import os
import io
import base64
import logging
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import httpx, fall back to requests
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    import requests
    HAS_HTTPX = False
    logger.warning("httpx not available, falling back to requests (synchronous only)")

# Local inference fallback imports
try:
    import torch
    import numpy as np
    import pandas as pd
    from PIL import Image
    from torchvision import transforms
    HAS_LOCAL_INFERENCE = True
except ImportError:
    HAS_LOCAL_INFERENCE = False


@dataclass
class ChutesConfig:
    """Configuration for Chutes API client."""
    api_key: str
    scene_chute_id: str = "scene-classifier-v1"
    timeout: float = 30.0
    max_retries: int = 2
    base_url: str = "https://api.chutes.ai/v1"

    @classmethod
    def from_env(cls) -> Optional["ChutesConfig"]:
        """Load configuration from environment variables."""
        api_key = os.environ.get("CHUTES_API_KEY")
        if not api_key:
            return None

        return cls(
            api_key=api_key,
            scene_chute_id=os.environ.get("CHUTES_SCENE_CHUTE_ID", "scene-classifier-v1"),
            timeout=float(os.environ.get("CHUTES_TIMEOUT", "30.0")),
            max_retries=int(os.environ.get("CHUTES_MAX_RETRIES", "2")),
            base_url=os.environ.get("CHUTES_BASE_URL", "https://api.chutes.ai/v1"),
        )


class ChutesSceneClassifier:
    """
    Chutes-based scene classifier with local fallback.

    This class wraps both Chutes remote inference and local PyTorch inference,
    using Chutes as primary (when configured) and falling back to local
    inference on network failures or if Chutes is not configured.
    """

    # Class mapping must match classify_scene.py
    CLASS_MAPPING = {
        'Screen Content / Text': 0,
        'Animation / Cartoon / Rendered Graphics': 1,
        'Faces / People': 2,
        'Gaming Content': 3,
        'other': 4,
        'unclear': 5
    }
    INV_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

    def __init__(
        self,
        config: Optional[ChutesConfig] = None,
        local_model_path: Optional[str] = None,
        device: str = "cpu",
        use_chutes: Optional[bool] = None
    ):
        """
        Initialize the classifier.

        Args:
            config: Chutes configuration. If None, loads from environment.
            local_model_path: Path to local model for fallback inference.
            device: Device for local inference ('cpu' or 'cuda').
            use_chutes: Override to force Chutes on/off. If None, uses USE_CHUTES env var.
        """
        self.config = config or ChutesConfig.from_env()
        self.local_model_path = local_model_path
        self.device = device
        self.http_client = None

        # Determine whether to use Chutes
        if use_chutes is not None:
            self.use_chutes = use_chutes
        else:
            self.use_chutes = os.environ.get("USE_CHUTES", "false").lower() == "true"

        # Initialize local model state (lazy-loaded)
        self._local_model = None
        self._local_metrics = None
        self._local_class_mapping = None
        self._local_scaler = None

        # Initialize HTTP client if using Chutes
        if self.use_chutes and self.config:
            self._init_http_client()

        logger.info(f"ChutesSceneClassifier initialized: use_chutes={self.use_chutes}, "
                   f"has_config={self.config is not None}, has_local_path={local_model_path is not None}")

    def _init_http_client(self):
        """Initialize HTTP client for Chutes API."""
        if not self.config:
            return

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        if HAS_HTTPX:
            self.http_client = httpx.AsyncClient(
                headers=headers,
                timeout=httpx.Timeout(self.config.timeout),
                follow_redirects=True
            )
        # For synchronous fallback with requests, we'll create session per-request

    def _encode_frame(self, frame_path: str) -> str:
        """Encode a frame as base64 string."""
        # Read and compress the image to reduce payload size
        try:
            from PIL import Image
            img = Image.open(frame_path).convert('RGB')
            # Resize to 224x224 to match model input (and reduce payload)
            img = img.resize((224, 224), Image.LANCZOS)

            # Encode to JPEG with quality 85 (balance size/quality)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85, optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return encoded
        except Exception as e:
            logger.error(f"Failed to encode frame {frame_path}: {e}")
            # Fallback: raw base64
            with open(frame_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')

    def _load_local_model(self):
        """Lazy-load local model for fallback inference."""
        if self._local_model is not None:
            return

        if not self.local_model_path or not os.path.exists(self.local_model_path):
            raise RuntimeError(f"Local model not available at {self.local_model_path}")

        if not HAS_LOCAL_INFERENCE:
            raise RuntimeError("Local inference dependencies not installed (torch, torchvision)")

        # Import the local classify_scene functions
        from .classify_scene import load_scene_classifier_model

        self._local_model, self._local_metrics, self._local_class_mapping = \
            load_scene_classifier_model(self.local_model_path, self.device)

        logger.info(f"Local model loaded with {len(self._local_metrics)} metrics")

    async def classify_async(
        self,
        frame_paths: List[str],
        video_features: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Classify a scene using Chutes (async).

        Args:
            frame_paths: List of paths to frames (typically 3 frames)
            video_features: Dictionary of video metrics

        Returns:
            Tuple of (label, detailed_results)
        """
        if self.use_chutes and self.config and self.http_client:
            try:
                return await self._classify_chutes_async(frame_paths, video_features)
            except Exception as e:
                logger.warning(f"Chutes inference failed: {e}, falling back to local")

        # Fallback to local inference
        return self._classify_local(frame_paths, video_features)

    async def _classify_chutes_async(
        self,
        frame_paths: List[str],
        video_features: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Make async Chutes API call."""
        # Encode frames
        frames_b64 = [self._encode_frame(p) for p in frame_paths]

        # Build payload - include video features and frames
        payload = {
            "frames": frames_b64,
            "video_features": video_features,
            "num_classes": 6
        }

        endpoint = f"{self.config.base_url}/chutes/{self.config.scene_chute_id}/run"

        for attempt in range(self.config.max_retries + 1):
            try:
                resp = await self.http_client.post(endpoint, json=payload)
                resp.raise_for_status()
                result = resp.json()

                return self._parse_chutes_response(result)

            except httpx.HTTPStatusError as e:
                logger.warning(f"Chutes API error (attempt {attempt + 1}): {e.response.status_code}")
                if attempt == self.config.max_retries:
                    raise
            except Exception as e:
                logger.warning(f"Chutes request error (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries:
                    raise

    def classify(
        self,
        frame_paths: List[str],
        video_features: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Classify a scene using Chutes (sync).

        Falls back to local inference if Chutes is unavailable.
        """
        if self.use_chutes and self.config:
            try:
                return self._classify_chutes_sync(frame_paths, video_features)
            except Exception as e:
                logger.warning(f"Chutes inference failed: {e}, falling back to local")

        # Fallback to local inference
        return self._classify_local(frame_paths, video_features)

    def _classify_chutes_sync(
        self,
        frame_paths: List[str],
        video_features: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Make synchronous Chutes API call using requests."""
        import requests

        # Encode frames
        frames_b64 = [self._encode_frame(p) for p in frame_paths]

        payload = {
            "frames": frames_b64,
            "video_features": video_features,
            "num_classes": 6
        }

        endpoint = f"{self.config.base_url}/chutes/{self.config.scene_chute_id}/run"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(self.config.max_retries + 1):
            try:
                resp = requests.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.config.timeout
                )
                resp.raise_for_status()
                result = resp.json()

                return self._parse_chutes_response(result)

            except requests.HTTPError as e:
                logger.warning(f"Chutes API error (attempt {attempt + 1}): {e.response.status_code if e.response else 'unknown'}")
                if attempt == self.config.max_retries:
                    raise
            except Exception as e:
                logger.warning(f"Chutes request error (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries:
                    raise

    def _parse_chutes_response(self, result: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Parse Chutes API response into standard format."""
        # The response format depends on the Chute implementation
        # Expected: {"label": str, "confidence": float, "probabilities": list}

        label = result.get("label", "unclear")
        confidence = result.get("confidence", 0.0)
        probs = result.get("probabilities", [0.0] * 6)

        # Ensure we have 6 probabilities
        if len(probs) < 6:
            probs = probs + [0.0] * (6 - len(probs))

        detailed_results = {
            'confidence_score': confidence,
            'prob_screen_content': float(probs[self.CLASS_MAPPING['Screen Content / Text']]),
            'prob_animation': float(probs[self.CLASS_MAPPING['Animation / Cartoon / Rendered Graphics']]),
            'prob_faces': float(probs[self.CLASS_MAPPING['Faces / People']]),
            'prob_gaming': float(probs[self.CLASS_MAPPING['Gaming Content']]),
            'prob_other': float(probs[self.CLASS_MAPPING['other']]),
            'prob_unclear': float(probs[self.CLASS_MAPPING['unclear']]),
            'frame_predictions': [label],  # Chutes handles frame aggregation
            'source': 'chutes'
        }

        return label, detailed_results

    def _classify_local(
        self,
        frame_paths: List[str],
        video_features: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Classify using local PyTorch model."""
        self._load_local_model()

        from .classify_scene import classify_scene_with_model, CombinedModel

        # Recreate the model from state dict
        model = CombinedModel(
            num_classes=len(self._local_class_mapping) if self._local_class_mapping else 6,
            model_type='mobilenet_v3_small',
            use_pretrained=False,
            metrics_dim=len(self._local_metrics)
        )
        model.load_state_dict(self._local_model)
        model.to(self.device)
        model.eval()

        label, detailed_results = classify_scene_with_model(
            frame_paths=frame_paths,
            video_features=video_features,
            scene_classifier=model,
            metrics_scaler=self._local_scaler,
            available_metrics=self._local_metrics,
            device=self.device,
            logging_enabled=False
        )

        detailed_results['source'] = 'local'
        return label, detailed_results

    async def close(self):
        """Close HTTP client if using async mode."""
        if self.http_client:
            await self.http_client.aclose()


# Convenience functions for backward compatibility with existing miner code

def load_scene_classifier_with_chutes(
    model_path: str,
    device: str = "cpu",
    use_chutes: Optional[bool] = None
) -> ChutesSceneClassifier:
    """
    Load scene classifier with Chutes support.

    This is the drop-in replacement for load_scene_classifier_model().
    It returns a ChutesSceneClassifier that can be used with classify_scene().
    """
    config = ChutesConfig.from_env()

    return ChutesSceneClassifier(
        config=config,
        local_model_path=model_path,
        device=device,
        use_chutes=use_chutes
    )


def classify_scene(
    frame_paths: List[str],
    video_features: Dict[str, Any],
    classifier: ChutesSceneClassifier
) -> Tuple[str, Dict[str, Any]]:
    """
    Classify a scene using the provided classifier.

    This is compatible with the existing classify_scene_with_model() interface
    but works with ChutesSceneClassifier objects.
    """
    return classifier.classify(frame_paths, video_features)


# Health check function for miner startup
def check_chutes_health() -> Dict[str, Any]:
    """
    Check if Chutes is properly configured and reachable.

    Returns a dict with status information:
        - configured: bool (API key present)
        - enabled: bool (USE_CHUTES=true)
        - reachable: bool (API is responding)
        - latency_ms: float (ping time if reachable)
    """
    result = {
        "configured": False,
        "enabled": False,
        "reachable": False,
        "latency_ms": None,
        "error": None
    }

    config = ChutesConfig.from_env()
    if not config:
        result["error"] = "CHUTES_API_KEY not set"
        return result

    result["configured"] = True
    result["enabled"] = os.environ.get("USE_CHUTES", "false").lower() == "true"

    # Try to reach the API
    import time
    try:
        import requests
        start = time.time()
        resp = requests.get(
            f"{config.base_url}/health",
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=10
        )
        elapsed = (time.time() - start) * 1000

        if resp.status_code == 200:
            result["reachable"] = True
            result["latency_ms"] = round(elapsed, 2)
        else:
            result["error"] = f"Health check returned {resp.status_code}"

    except Exception as e:
        result["error"] = str(e)

    return result
