"""Voice Enhancement Module using Resemble-Enhance and VoiceFixer."""

import torch
import numpy as np
from typing import Literal, Optional
import logging

logger = logging.getLogger(__name__)


class VoiceEnhancer:
    """
    Voice enhancement using Resemble-Enhance (primary) or VoiceFixer (fallback).

    Enhances separated speech quality and removes residual artifacts.
    """

    def __init__(
        self,
        model_type: Literal["resemble", "voicefixer"] = "resemble",
        device: str = "cuda",
        **kwargs
    ):
        """
        Initialize voice enhancer.

        Args:
            model_type: Enhancement model to use
            device: Device for processing (cuda/cpu)
            **kwargs: Model-specific configuration
        """
        self.model_type = model_type
        self.device = device
        self.config = kwargs
        self.model = None

    def load_model(self) -> None:
        """Load enhancement model."""
        if self.model is not None:
            logger.info(f"Model already loaded: {self.model_type}")
            return

        if self.model_type == "resemble":
            self._load_resemble_model()
        elif self.model_type == "voicefixer":
            self._load_voicefixer_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _load_resemble_model(self) -> None:
        """Load Resemble-Enhance model."""
        try:
            # Note: Resemble-Enhance installation:
            # pip install git+https://github.com/resemble-ai/resemble-enhance.git
            # Resemble-Enhance uses functional API, no model loading needed at init
            logger.info("Resemble-Enhance API ready")
            self.model = "resemble"  # Marker that model type is selected

        except ImportError:
            logger.error(
                "Resemble-Enhance not installed. "
                "Install with: pip install git+https://github.com/resemble-ai/resemble-enhance.git"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load Resemble-Enhance: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def _load_voicefixer_model(self) -> None:
        """Load VoiceFixer model."""
        try:
            from voicefixer import VoiceFixer as VF  # type: ignore[import-untyped]

            logger.info("Loading VoiceFixer model...")

            mode = self.config.get('mode', 2)
            cuda = self.config.get('cuda', torch.cuda.is_available())

            self.model = VF()
            self.voicefixer_mode = mode
            self.voicefixer_cuda = cuda

            logger.info("VoiceFixer model loaded successfully")

        except ImportError:
            logger.error("VoiceFixer not installed. Install with: pip install voicefixer")
            raise
        except Exception as e:
            logger.error(f"Failed to load VoiceFixer: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def enhance_speech(
        self,
        vocals: np.ndarray,
        sr: int,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Enhance speech quality.

        Args:
            vocals: Vocals waveform
            sr: Sample rate
            output_path: Optional path to save enhanced audio

        Returns:
            Enhanced vocals waveform
        """
        if self.model is None:
            self.load_model()

        logger.info(f"Enhancing speech using {self.model_type}")

        try:
            if self.model_type == "resemble":
                enhanced = self._enhance_with_resemble(vocals, sr, output_path)
            elif self.model_type == "voicefixer":
                enhanced = self._enhance_with_voicefixer(vocals, sr, output_path)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            logger.info("Enhancement complete")
            return enhanced

        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            raise RuntimeError(f"Voice enhancement failed: {e}")

    def _enhance_with_resemble(
        self,
        vocals: np.ndarray,
        sr: int,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """Enhance using Resemble-Enhance."""
        import torch
        import os

        # Set CUDA_HOME to prevent compilation attempts
        if 'CUDA_HOME' not in os.environ:
            # Create a mock CUDA_HOME to bypass compilation checks
            os.environ['CUDA_HOME'] = os.path.expanduser('~/.local/cuda-mock')
            logger.info(f"Set CUDA_HOME={os.environ['CUDA_HOME']}")

        try:
            from resemble_enhance.enhancer.inference import enhance  # type: ignore[import-untyped]
        except RuntimeError as e:
            if "CUDA_HOME" in str(e):
                logger.error(f"Resemble-Enhance CUDA compilation failed: {e}")
                logger.info("Falling back to CPU processing")
                self.device = "cpu"
                from resemble_enhance.enhancer.inference import enhance  # type: ignore[import-untyped]
            else:
                raise

        # Get configuration
        nfe = self.config.get('nfe', 64)
        solver = self.config.get('solver', 'midpoint')

        logger.info(f"Running Resemble-Enhance with nfe={nfe}, solver={solver}, device={self.device}")

        # Convert to torch tensor
        dwav_torch = torch.from_numpy(vocals).float()

        # Ensure it's 1D
        if dwav_torch.ndim > 1:
            dwav_torch = dwav_torch.squeeze()

        # Run enhancement with error handling
        try:
            enhanced_torch, new_sr = enhance(
                dwav=dwav_torch,
                sr=sr,
                device=self.device,
                nfe=nfe,
                solver=solver,
                run_dir=None  # Will auto-download model
            )
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                logger.warning(f"CUDA error during enhancement: {e}")
                logger.info("Retrying with CPU")
                self.device = "cpu"
                enhanced_torch, new_sr = enhance(
                    dwav=dwav_torch,
                    sr=sr,
                    device="cpu",
                    nfe=nfe,
                    solver=solver,
                    run_dir=None
                )
            else:
                raise

        # Convert back to numpy
        enhanced = enhanced_torch.cpu().numpy()

        # Save if output_path specified
        if output_path is not None:
            import soundfile as sf  # type: ignore[import-untyped]
            sf.write(output_path, enhanced, new_sr)

        return enhanced

    def _enhance_with_voicefixer(
        self,
        vocals: np.ndarray,
        sr: int,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """Enhance using VoiceFixer."""
        import soundfile as sf  # type: ignore[import-untyped]
        import tempfile
        import os

        # VoiceFixer works with file paths
        # Create temporary input file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_in:
            sf.write(tmp_in.name, vocals, sr)
            tmp_in_path = tmp_in.name

        # Create temporary output file if not specified
        if output_path is None:
            tmp_out = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = tmp_out.name
            tmp_out.close()

        try:
            # Run VoiceFixer restoration
            self.model.restore(
                input=tmp_in_path,
                output=output_path,
                cuda=self.voicefixer_cuda,
                mode=self.voicefixer_mode
            )

            # Load enhanced audio
            enhanced, _ = sf.read(output_path)

            return enhanced

        finally:
            # Cleanup temporary files
            if os.path.exists(tmp_in_path):
                os.remove(tmp_in_path)

    def remove_artifacts(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove artifacts from audio.

        Args:
            audio: Audio waveform

        Returns:
            Cleaned audio
        """
        # Simple artifact removal using spectral gating
        # This is a placeholder - actual implementation would be more sophisticated
        return audio

    def upscale_bandwidth(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """
        Upscale audio bandwidth.

        Args:
            audio: Audio waveform
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Upscaled audio
        """
        import librosa  # type: ignore[import-untyped]

        if orig_sr >= target_sr:
            return audio

        logger.info(f"Upscaling bandwidth from {orig_sr}Hz to {target_sr}Hz")
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
