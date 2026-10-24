"""
YOLOv10 Model Saving and Export

Handles downloading, saving, and exporting YOLOv10 models in various formats.
"""

import logging
import warnings
from pathlib import Path

from absl import app, flags
from absl.flags import FLAGS

from yolov10.config import YOLOv10Config
from yolov10.model_loader import ModelLoader

# Suppress ultralytics torch.load FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="ultralytics.*")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Command line flags
flags.DEFINE_string("model", "yolov10s", "YOLOv10 model variant (n, s, m, b, l, x)")
flags.DEFINE_string(
    "weights", None, "path to custom weights file (if not using pretrained)"
)
flags.DEFINE_string("output", "./checkpoints/yolov10", "path to output directory")
flags.DEFINE_string(
    "export_format", "onnx", "export format (onnx, torchscript, coreml, tflite)"
)
flags.DEFINE_string("device", "auto", "device to use (auto, cpu, cuda, mps)")
flags.DEFINE_float("confidence", 0.25, "confidence threshold for detection")
flags.DEFINE_float("iou", 0.45, "IOU threshold for NMS")
flags.DEFINE_boolean("optimize", True, "optimize model for inference")
flags.DEFINE_boolean("half", True, "use half precision (FP16) for GPU")
flags.DEFINE_integer("img_size", 640, "input image size")


def save_yolov10_model():
    """
    Save and export YOLOv10 model.

    This function downloads/loads a YOLOv10 model and exports it in the specified format.
    """
    logger.info("Starting YOLOv10 model saving process...")

    # Create output directory
    output_path = Path(FLAGS.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize YOLOv10 configuration
    config = YOLOv10Config(
        model_variant=FLAGS.model,
        model_path=FLAGS.weights,
        confidence_threshold=FLAGS.confidence,
        iou_threshold=FLAGS.iou,
        device=FLAGS.device,
        input_size=FLAGS.img_size,
        half_precision=FLAGS.half,
        export_format=FLAGS.export_format,
        optimize_for_speed=FLAGS.optimize,
        verbose=True,
    )

    logger.info(f"Configuration: {config.to_dict()}")

    # Validate configuration
    config.validate()

    # Initialize model loader
    model_loader = ModelLoader(
        model_variant=config.model_variant,
        device=config.device,
        confidence_threshold=config.confidence_threshold,
        iou_threshold=config.iou_threshold,
    )

    try:
        # Load model
        logger.info(f"Loading YOLOv10 model: {config.model_variant}")
        model = model_loader.load_model(FLAGS.weights)

        # Optimize model
        if FLAGS.optimize:
            logger.info("Optimizing model for inference...")
            model_loader.optimize_for_inference()

        # Get model information
        model_info = model_loader.get_model_info()
        logger.info("Model loaded successfully:")
        logger.info(f"  Variant: {model_info['variant']}")
        logger.info(f"  Device: {model_info['device']}")
        logger.info(f"  Classes: {len(model_info['model_names'])}")

        # Export model
        logger.info(f"Exporting model to {FLAGS.export_format} format...")
        export_path = model_loader.export_model(
            export_format=FLAGS.export_format,
            export_path=str(
                output_path / f"{config.model_variant}.{FLAGS.export_format}"
            ),
        )

        logger.info(f"Model exported successfully to: {export_path}")

        # Save PyTorch model as well
        pytorch_path = output_path / f"{config.model_variant}.pt"
        if hasattr(model.model, "state_dict"):
            import torch

            torch.save(model.model.state_dict(), pytorch_path)
            logger.info(f"PyTorch model saved to: {pytorch_path}")

        # Save configuration
        import json

        config_path = output_path / f"{config.model_variant}_config.json"
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to: {config_path}")

        logger.info("YOLOv10 model saving completed successfully!")

        return export_path

    except Exception as e:
        logger.error(f"Error during model saving: {e}")
        raise


def download_model_variants():
    """
    Download all YOLOv10 model variants for testing.
    """
    logger.info("Downloading all YOLOv10 model variants...")

    variants = ["yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x"]
    output_path = Path(FLAGS.output)

    for variant in variants:
        try:
            logger.info(f"Downloading {variant}...")
            model_loader = ModelLoader(model_variant=variant)
            model = model_loader.load_model()

            # Save to checkpoint
            variant_path = output_path / f"{variant}.pt"
            model.model.save(variant_path)
            logger.info(f"Saved {variant} to {variant_path}")

        except Exception as e:
            logger.error(f"Failed to download {variant}: {e}")

    logger.info("Model download completed!")


def benchmark_models():
    """
    Benchmark different YOLOv10 model variants.
    """
    logger.info("Benchmarking YOLOv10 model variants...")

    variants = ["yolov10n", "yolov10s", "yolov10m"]

    for variant in variants:
        try:
            logger.info(f"Benchmarking {variant}...")

            config = YOLOv10Config(model_variant=variant, verbose=True)
            model_loader = ModelLoader(model_variant=variant)
            model = model_loader.load_model()

            # Get model info
            model_info = model_loader.get_model_info()
            model_specs = config.get_model_info()

            logger.info(f"{variant} specs:")
            logger.info(f"  Parameters: {model_specs.get('params', 'Unknown')}")
            logger.info(f"  FLOPS: {model_specs.get('flops', 'Unknown')}")
            logger.info(f"  mAP: {model_specs.get('map', 'Unknown')}")
            logger.info(f"  Device: {model_info['device']}")

        except Exception as e:
            logger.error(f"Failed to benchmark {variant}: {e}")


def main(_argv):
    """
    Main function for save_model.py
    """
    logger.info("YOLOv10 Model Saving Tool")
    logger.info(f"Output directory: {FLAGS.output}")
    logger.info(f"Model variant: {FLAGS.model}")
    logger.info(f"Export format: {FLAGS.export_format}")

    # Save/export the main model
    export_path = save_yolov10_model()

    # Optional: Download all variants if requested
    if hasattr(FLAGS, "download_all") and FLAGS.download_all:
        download_model_variants()

    # Optional: Benchmark models if requested
    if hasattr(FLAGS, "benchmark") and FLAGS.benchmark:
        benchmark_models()

    logger.info("All operations completed successfully!")


if __name__ == "__main__":
    # Add optional flags
    flags.DEFINE_boolean("download_all", False, "download all YOLOv10 model variants")
    flags.DEFINE_boolean("benchmark", False, "benchmark different model variants")

    try:
        app.run(main)
    except SystemExit:
        pass
