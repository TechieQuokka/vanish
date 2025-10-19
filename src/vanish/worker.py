"""Worker processes for memory-efficient task execution."""

import argparse
import sys
import numpy as np
import soundfile as sf
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def worker_separate(input_file: str, output_vocals: str, output_background: str, model: str, device: str):
    """Worker: Source separation."""
    from vanish.modules import SourceSeparator

    logger.info(f"Worker: Separating {input_file}")

    # Load audio
    audio, sr = sf.read(input_file)

    # Initialize separator
    separator = SourceSeparator(model_name=model, device=device)

    # Separate
    separated = separator.separate(audio, sr)

    # Save outputs
    sf.write(output_vocals, separated.vocals, sr)
    sf.write(output_background, separated.background, sr)

    logger.info(f"Separation complete: {output_vocals}, {output_background}")


def worker_enhance(input_file: str, output_file: str, model_type: str, device: str, **kwargs):
    """Worker: Voice enhancement."""
    from vanish.modules import VoiceEnhancer

    logger.info(f"Worker: Enhancing {input_file} with {model_type}")

    # Load audio
    audio, sr = sf.read(input_file)

    # Initialize enhancer
    enhancer = VoiceEnhancer(model_type=model_type, device=device, **kwargs)

    # Enhance
    enhanced = enhancer.enhance_speech(audio, sr)

    # Save output
    sf.write(output_file, enhanced, sr)

    logger.info(f"Enhancement complete: {output_file}")


def worker_postprocess(input_file: str, output_file: str, **kwargs):
    """Worker: Post-processing."""
    from vanish.modules import PostProcessor

    logger.info(f"Worker: Post-processing {input_file}")

    # Load audio
    audio, sr = sf.read(input_file)

    # Initialize post-processor
    processor = PostProcessor(**kwargs)

    # Process
    processed = processor.process(audio, sr)

    # Save output
    sf.write(output_file, processed, sr)

    logger.info(f"Post-processing complete: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio processing worker")
    parser.add_argument("task", choices=["separate", "enhance", "postprocess"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output2", help="Second output (for separation)")
    parser.add_argument("--model", default="htdemucs")
    parser.add_argument("--model-type", default="resemble")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kwargs", help="JSON string of kwargs")

    args = parser.parse_args()

    # Parse kwargs if provided
    kwargs = {}
    if args.kwargs:
        import json
        kwargs = json.loads(args.kwargs)

    try:
        if args.task == "separate":
            worker_separate(args.input, args.output, args.output2, args.model, args.device)
        elif args.task == "enhance":
            worker_enhance(args.input, args.output, args.model_type, args.device, **kwargs)
        elif args.task == "postprocess":
            worker_postprocess(args.input, args.output, **kwargs)

        sys.exit(0)

    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
        sys.exit(1)
