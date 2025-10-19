"""Main audio processing pipeline orchestration."""

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
import logging
import subprocess
import sys
import json

from vanish.config import PipelineConfig
from vanish.modules import (
    AudioInput,
    SourceSeparator,
    VoiceEnhancer,
    PostProcessor,
    QualityAssessor
)
from vanish.modules.quality_assessor import QualityMetrics

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Container for processing results."""

    output_path: str
    metrics: QualityMetrics
    processing_time: float
    intermediate_files: Optional[dict] = None


class AudioPipeline:
    """
    Complete audio noise removal pipeline.

    Pipeline stages:
    1. Audio input and validation
    2. Source separation (Demucs)
    3. Voice enhancement (Resemble-Enhance/VoiceFixer)
    4. Post-processing
    5. Quality assessment
    """

    def __init__(self, config: Optional[PipelineConfig] = None, use_workers: bool = True):
        """
        Initialize audio pipeline.

        Args:
            config: Pipeline configuration (uses defaults if None)
            use_workers: Use separate worker processes for memory efficiency
        """
        if config is None:
            config = PipelineConfig()

        config.validate()
        self.config = config
        self.use_workers = use_workers

        # Initialize lightweight modules (no heavy models)
        self.audio_input = AudioInput(target_sr=config.target_sr)

        # For quality assessment only (lightweight)
        self.quality_assessor = QualityAssessor(
            calculate_snr=config.quality.calculate_snr,
            calculate_pesq=config.quality.calculate_pesq,
            calculate_stoi=config.quality.calculate_stoi,
        )

        # Heavy models will be loaded in worker processes
        if not use_workers:
            # Legacy mode: load everything in main process
            self.source_separator = SourceSeparator(
                model_name=config.demucs.model,
                device=config.demucs.device,
                shifts=config.demucs.shifts,
                overlap=config.demucs.overlap,
            )

            # Initialize enhancer based on mode
            if config.enhancement_mode in ["resemble", "both"]:
                self.enhancer_primary = VoiceEnhancer(
                    model_type="resemble",
                    device=config.resemble.device,
                    denoiser_run_steps=config.resemble.denoiser_run_steps,
                    enhance_run_steps=config.resemble.enhance_run_steps,
                    solver=config.resemble.solver,
                    nfe=config.resemble.nfe,
                )
            else:
                self.enhancer_primary = None

            if config.enhancement_mode in ["voicefixer", "both"]:
                self.enhancer_fallback = VoiceEnhancer(
                    model_type="voicefixer",
                    device=config.voicefixer.mode,
                    mode=config.voicefixer.mode,
                    cuda=config.voicefixer.cuda,
                )
            else:
                self.enhancer_fallback = None

            self.post_processor = PostProcessor(
                noise_gate_threshold=config.postprocess.noise_gate_threshold,
                target_lufs=config.postprocess.target_lufs,
                highpass_cutoff=config.postprocess.highpass_cutoff,
                apply_deessing=config.postprocess.apply_deessing,
                deess_frequency=config.postprocess.deess_frequency,
                deess_threshold=config.postprocess.deess_threshold,
            )
        else:
            self.source_separator = None
            self.enhancer_primary = None
            self.enhancer_fallback = None
            self.post_processor = None

        logger.info(f"Pipeline initialized (worker mode: {use_workers})")

    def _run_worker(self, task: str, **kwargs) -> int:
        """Run worker process and return exit code."""
        worker_script = Path(__file__).parent / "worker.py"

        cmd = [sys.executable, str(worker_script), task]

        for key, value in kwargs.items():
            if value is not None:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        logger.info(f"Running worker: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.stdout:
            logger.info(f"Worker stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Worker stderr:\n{result.stderr}")

        return result.returncode

    def process(
        self,
        input_path: str,
        output_path: str,
        save_intermediate: Optional[bool] = None,
    ) -> ProcessingResult:
        """
        Process audio file through complete pipeline.

        Args:
            input_path: Path to input audio file
            output_path: Path for output audio file
            save_intermediate: Whether to save intermediate files

        Returns:
            ProcessingResult with metrics and paths
        """
        import time

        start_time = time.time()

        logger.info(f"Starting processing: {input_path} -> {output_path}")

        # Determine if we should save intermediate files
        if save_intermediate is None:
            save_intermediate = self.config.save_intermediate

        intermediate_files = {} if save_intermediate else None
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Stage 1: Audio input and preprocessing
            logger.info("Stage 1: Audio input and preprocessing")
            audio, sr, quality_metrics = self.audio_input.preprocess(input_path)

            preprocessed_path = output_dir / "01_preprocessed.wav"
            sf.write(preprocessed_path, audio, sr)
            if save_intermediate:
                intermediate_files['preprocessed'] = str(preprocessed_path)

            # Store original for quality comparison
            original_audio = audio.copy()

            # Stage 2: Source separation (worker process)
            logger.info("Stage 2: Source separation (Demucs) - Worker Process")
            vocals_path = output_dir / "02_vocals_separated.wav"
            background_path = output_dir / "02_background.wav"

            if self.use_workers:
                returncode = self._run_worker(
                    "separate",
                    input=str(preprocessed_path),
                    output=str(vocals_path),
                    output2=str(background_path),
                    model=self.config.demucs.model,
                    device=self.config.demucs.device
                )

                if returncode != 0:
                    raise RuntimeError(f"Separation worker failed with code {returncode}")
            else:
                # Legacy mode
                separated = self.source_separator.separate(audio, sr)
                sf.write(vocals_path, separated.vocals, sr)
                sf.write(background_path, separated.background, sr)

            if save_intermediate:
                intermediate_files['vocals'] = str(vocals_path)
                intermediate_files['background'] = str(background_path)

            # Stage 3: Voice enhancement (worker process)
            logger.info("Stage 3: Voice enhancement - Worker Process")
            enhanced_path = output_dir / "03_enhanced.wav"

            if self.use_workers:
                # Build kwargs JSON for worker
                if self.config.enhancement_mode in ["resemble", "both"]:
                    kwargs = {
                        "nfe": self.config.resemble.nfe,
                        "solver": self.config.resemble.solver
                    }
                    returncode = self._run_worker(
                        "enhance",
                        input=str(vocals_path),
                        output=str(enhanced_path),
                        model_type="resemble",
                        device=self.config.resemble.device,
                        kwargs=json.dumps(kwargs)
                    )

                    if returncode != 0:
                        logger.warning(f"Enhancement worker failed with code {returncode}, using vocals")
                        # Copy vocals as fallback
                        import shutil
                        shutil.copy(vocals_path, enhanced_path)
                else:
                    # No enhancement, just copy
                    import shutil
                    shutil.copy(vocals_path, enhanced_path)
            else:
                # Legacy mode
                vocals, _ = sf.read(vocals_path)
                enhanced = self._enhance_vocals(vocals, sr, output_dir, False)
                if enhanced is None:
                    enhanced = vocals
                sf.write(enhanced_path, enhanced, sr)

            if save_intermediate:
                intermediate_files['enhanced'] = str(enhanced_path)

            # Stage 4: Post-processing (worker process)
            logger.info("Stage 4: Post-processing - Worker Process")
            processed_path = output_dir / "04_postprocessed.wav"

            if self.use_workers:
                kwargs = {
                    "noise_gate_threshold": self.config.postprocess.noise_gate_threshold,
                    "target_lufs": self.config.postprocess.target_lufs,
                    "highpass_cutoff": self.config.postprocess.highpass_cutoff,
                    "apply_deessing": self.config.postprocess.apply_deessing,
                    "deess_frequency": self.config.postprocess.deess_frequency,
                    "deess_threshold": self.config.postprocess.deess_threshold,
                }

                returncode = self._run_worker(
                    "postprocess",
                    input=str(enhanced_path),
                    output=str(processed_path),
                    kwargs=json.dumps(kwargs)
                )

                if returncode != 0:
                    raise RuntimeError(f"Post-processing worker failed with code {returncode}")
            else:
                # Legacy mode
                enhanced, _ = sf.read(enhanced_path)
                processed = self.post_processor.process(enhanced, sr)
                sf.write(processed_path, processed, sr)

            if save_intermediate:
                intermediate_files['postprocessed'] = str(processed_path)

            # Stage 5: Save final output
            logger.info(f"Stage 5: Saving final output to {output_path}")
            output_path = Path(output_path)

            # Load processed audio
            processed, sr = sf.read(processed_path)

            # Determine bit depth
            subtype = f'PCM_{self.config.output_bitdepth}'

            sf.write(output_path, processed, sr, subtype=subtype)

            # Stage 6: Quality assessment (simplified for long audio)
            logger.info("Stage 6: Quality assessment")
            try:
                # For very long audio, skip quality assessment to avoid memory issues
                duration = len(processed) / sr
                if duration > 300:  # > 5 minutes
                    logger.warning(f"Audio too long ({duration:.1f}s), skipping quality assessment")
                    from vanish.modules.quality_assessor import QualityMetrics
                    metrics = QualityMetrics()
                    metrics.snr = None
                    metrics.pesq = None
                    metrics.stoi = None
                else:
                    metrics = self.quality_assessor.assess(processed, original_audio, sr)
            except Exception as e:
                logger.warning(f"Quality assessment failed: {e}, creating dummy metrics")
                from vanish.modules.quality_assessor import QualityMetrics
                metrics = QualityMetrics()
                metrics.snr = None
                metrics.pesq = None
                metrics.stoi = None

            # Processing time
            processing_time = time.time() - start_time

            logger.info(f"Processing complete in {processing_time:.2f}s")
            if metrics.snr is not None and metrics.snr > 0:
                logger.info(f"\n{metrics}")

            return ProcessingResult(
                output_path=str(output_path),
                metrics=metrics,
                processing_time=processing_time,
                intermediate_files=intermediate_files,
            )

        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            raise RuntimeError(f"Pipeline processing failed: {e}")

    def _enhance_vocals(
        self,
        vocals: np.ndarray,
        sr: int,
        output_dir: Path,
        save_intermediate: bool
    ) -> Optional[np.ndarray]:
        """
        Enhance vocals using configured enhancement mode.

        Args:
            vocals: Vocals waveform
            sr: Sample rate
            output_dir: Directory for intermediate files
            save_intermediate: Whether to save intermediate files

        Returns:
            Enhanced vocals or None if enhancement fails
        """
        try:
            # Try primary enhancer (Resemble-Enhance)
            if self.enhancer_primary is not None:
                logger.info("Enhancing with Resemble-Enhance")
                enhanced = self.enhancer_primary.enhance_speech(vocals, sr)
                return enhanced

            # Try fallback enhancer (VoiceFixer)
            if self.enhancer_fallback is not None:
                logger.info("Enhancing with VoiceFixer")
                enhanced = self.enhancer_fallback.enhance_speech(vocals, sr)
                return enhanced

            # No enhancer available
            logger.warning("No voice enhancer configured")
            return vocals

        except Exception as e:
            logger.error(f"Voice enhancement failed: {e}")

            # Try fallback if primary failed
            if self.enhancer_primary is not None and self.enhancer_fallback is not None:
                try:
                    logger.info("Trying fallback enhancer (VoiceFixer)")
                    enhanced = self.enhancer_fallback.enhance_speech(vocals, sr)
                    return enhanced
                except Exception as e2:
                    logger.error(f"Fallback enhancement also failed: {e2}")

            return None

    def process_batch(
        self,
        input_files: list[str],
        output_dir: str,
        prefix: str = "clean_"
    ) -> list[ProcessingResult]:
        """
        Process multiple audio files in batch.

        Args:
            input_files: List of input file paths
            output_dir: Output directory
            prefix: Prefix for output filenames

        Returns:
            List of ProcessingResult objects
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = []

        for i, input_file in enumerate(input_files, 1):
            logger.info(f"Processing file {i}/{len(input_files)}: {input_file}")

            try:
                # Generate output filename
                input_path = Path(input_file)
                output_file = output_path / f"{prefix}{input_path.name}"

                # Process file
                result = self.process(str(input_file), str(output_file))
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to process {input_file}: {e}")
                # Continue with next file
                continue

        logger.info(f"Batch processing complete: {len(results)}/{len(input_files)} successful")

        return results
