"""
This script gives us a good understanding of the OCR model we're using
to train/fine-tune the LLM since the facility blueprints (if available)
should be integrated with the LLM to give it good context around the
HVAC operations.
"""

import os
import gc
import json
import logging
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoProcessor,
    AutoModel,
    VisionEncoderDecoderModel,
    TrOCRProcessor,
)
import matplotlib.pyplot as plt
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class VisionModelValidator:
    """
    Validate either the Vision or the OCR model
    """

    def __init__(self, model_id, model_type=None, use_cache=True):
        """
        Initialize the Vision/OCR Model Validator.

        Args:
            model_id: HuggingFace model ID
            model_type: Either "trocr" or "mllama" - will be auto-detected if None
            use_cache: Whether to use cached models
        """
        self.model_id = model_id
        self.model_type = model_type
        self.use_cache = use_cache
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

    def detect_model_type(self):
        """Detect the model type based on the config file."""
        try:
            # Try to load config
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(self.model_id)

            # Check model type
            if hasattr(config, "model_type"):
                if config.model_type == "vision-encoder-decoder" or config.model_type == "trocr":
                    self.model_type = "trocr"
                    logger.info("Detected model type: TrOCR (Vision-Encoder-Decoder)")
                elif config.model_type == "mllama":
                    self.model_type = "mllama"
                    logger.info("Detected model type: Llama Vision")
                else:
                    logger.warning(
                        "Unknown model type: %s. Will try to infer from architecture.",
                        config.model_type,
                    )

            self.config = config
            return self.model_type is not None
        except Exception as e:
            logger.error("Error detecting model type: %s", e)
            return False

    def load_model(self):
        """Load the model, processor, and tokenizer based on the model type."""
        try:
            # Auto-detect model type if not provided
            if self.model_type is None:
                if not self.detect_model_type():
                    logger.error("Could not detect model type automatically.")
                    return False

            # Load appropriate model and processor based on model type
            if self.model_type == "trocr":
                logger.info("Loading TrOCR model from %s", self.model_id)
                self.processor = TrOCRProcessor.from_pretrained(self.model_id)
                self.model = VisionEncoderDecoderModel.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    use_cache=self.use_cache,
                )
            elif self.model_type == "mllama":
                logger.info("Loading Llama Vision model from %s", self.model_id)
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = AutoModel.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    use_cache=self.use_cache,
                )
            else:
                logger.error("Unsupported model type: %s", self.model_type)
                return False

            logger.info("Model and processor loaded successfully")
            return True
        except Exception as e:
            logger.error("Error loading model: %s", e)
            return False

    def validate_model_architecture(self):
        """Validate the model architecture based on the model type."""
        if self.model is None:
            logger.error("Model not loaded")
            return False

        try:
            # Get model configuration
            config = self.model.config if self.config is None else self.config

            # Log key configuration parameters based on model type
            logger.info("Model architecture validation:")

            if self.model_type == "trocr":
                # Check encoder (Vision) configuration
                if hasattr(config, "encoder") and config.encoder:
                    logger.info(
                        "  Vision Encoder type: %s", config.encoder.get("model_type", "unknown")
                    )
                    logger.info(
                        "  Vision Hidden size: %s", config.encoder.get("hidden_size", "unknown")
                    )
                    logger.info(
                        "  Vision Number of layers: %s",
                        config.encoder.get("num_hidden_layers", "unknown"),
                    )
                    logger.info(
                        "  Vision Number of attention heads: %s",
                        config.encoder.get("num_attention_heads", "unknown"),
                    )
                    logger.info(
                        "  Vision Image size: %s", config.encoder.get("image_size", "unknown")
                    )

                # Check decoder (Text) configuration
                if hasattr(config, "decoder") and config.decoder:
                    logger.info(
                        "  Text Decoder type: %s", config.decoder.get("model_type", "unknown")
                    )
                    logger.info("  Text Hidden size: %s", config.decoder.get("d_model", "unknown"))
                    logger.info(
                        "  Text Number of layers: %s",
                        config.decoder.get("decoder_layers", "unknown"),
                    )
                    logger.info(
                        "  Text Number of attention heads: %s",
                        config.decoder.get("decoder_attention_heads", "unknown"),
                    )
                    logger.info(
                        "  Text Vocab size: %s", config.decoder.get("vocab_size", "unknown")
                    )

            elif self.model_type == "mllama":
                # Check vision configuration
                if hasattr(config, "vision_config") and config.vision_config:
                    vision_config = config.vision_config
                    logger.info(
                        "  Vision model type: %s", vision_config.get("model_type", "unknown")
                    )
                    logger.info(
                        "  Vision hidden size: %s", vision_config.get("hidden_size", "unknown")
                    )
                    logger.info(
                        "  Vision layers: %s", vision_config.get("num_hidden_layers", "unknown")
                    )
                    logger.info(
                        "  Vision attention heads: %s",
                        vision_config.get("attention_heads", "unknown"),
                    )
                    logger.info(
                        "  Vision image size: %s", vision_config.get("image_size", "unknown")
                    )
                    logger.info(
                        "  Vision patch size: %s", vision_config.get("patch_size", "unknown")
                    )

                # Check text configuration
                if hasattr(config, "text_config") and config.text_config:
                    text_config = config.text_config
                    logger.info("  Text model type: %s", text_config.get("model_type", "unknown"))
                    logger.info("  Text hidden size: %s", text_config.get("hidden_size", "unknown"))
                    logger.info(
                        "  Text layers: %s", text_config.get("num_hidden_layers", "unknown")
                    )
                    logger.info(
                        "  Text attention heads: %s",
                        text_config.get("num_attention_heads", "unknown"),
                    )
                    logger.info("  Text vocab size: %s", text_config.get("vocab_size", "unknown"))

            # Calculate and validate model parameter count
            model_parameters = sum(p.numel() for p in self.model.parameters())
            logger.info("  Total parameters: %s", f"{model_parameters:,}")

            # Roughly check if parameter count makes sense
            if self.model_type == "trocr" and model_parameters < 1e8:
                logger.warning(
                    "TrOCR model has fewer parameters than expected (%s)", f"{model_parameters:,}"
                )
            elif self.model_type == "mllama" and model_parameters < 1e9:
                logger.warning(
                    "Llama Vision model has fewer parameters than expected (%s)",
                    f"{model_parameters:,}",
                )

            return True
        except Exception as e:
            logger.error("Error validating model architecture: %s", e)
            return False

    def validate_tensor_properties(self):
        """Validate tensor properties such as shapes, dtypes, and numerical properties."""
        if self.model is None:
            logger.error("Model not loaded")
            return False

        try:
            # Collect tensor statistics
            tensor_stats = {
                "dtypes": {},
                "has_nan": False,
                "has_inf": False,
                "weight_norms": [],
                "layer_names": [],
            }

            # Examine a subset of model parameters for efficiency
            subset_size = 20  # Examine only a subset of layers for efficiency
            layer_names = [name for name, _ in self.model.named_parameters()]
            sample_layers = (
                layer_names[:subset_size] if len(layer_names) > subset_size else layer_names
            )

            for name, param in tqdm(
                [(n, p) for n, p in self.model.named_parameters() if n in sample_layers],
                desc="Validating tensor properties",
            ):
                # Check data types
                if str(param.dtype) not in tensor_stats["dtypes"]:
                    tensor_stats["dtypes"][str(param.dtype)] = 0
                tensor_stats["dtypes"][str(param.dtype)] += 1

                # Check for NaNs and Infs in weights
                if torch.isnan(param).any():
                    tensor_stats["has_nan"] = True
                    logger.warning("NaN values detected in %s", name)

                if torch.isinf(param).any():
                    tensor_stats["has_inf"] = True
                    logger.warning("Inf values detected in %s", name)

                # Calculate weight norm
                norm = torch.norm(param).item()
                tensor_stats["weight_norms"].append(norm)
                tensor_stats["layer_names"].append(name)

            # Check for consistent dtype usage
            logger.info("Data type distribution: %s", tensor_stats["dtypes"])
            if len(tensor_stats["dtypes"]) > 2:  # Allowing for mixed precision
                logger.warning("Model uses multiple data types: %s", tensor_stats["dtypes"])

            # Report NaN and Inf findings
            if tensor_stats["has_nan"]:
                logger.error("Model contains NaN values!")
            else:
                logger.info("No NaN values detected in examined parameters")

            if tensor_stats["has_inf"]:
                logger.error("Model contains Inf values!")
            else:
                logger.info("No Inf values detected in examined parameters")

            # Plot weight norms for visual inspection
            if len(tensor_stats["weight_norms"]) > 0:
                plt.figure(figsize=(10, 6))
                plt.bar(range(len(tensor_stats["weight_norms"])), tensor_stats["weight_norms"])
                plt.xlabel("Layer index")
                plt.ylabel("Weight norm")
                plt.title("Weight norms across model layers")
                plt.tight_layout()
                plt.savefig("weight_norms.png")
                logger.info("Weight norm distribution saved to weight_norms.png")

            return not (tensor_stats["has_nan"] or tensor_stats["has_inf"])
        except Exception as e:
            logger.error("Error validating tensor properties: %s", e)
            return False

    def validate_sample_image(self, image_path=None):
        """Test the model with a sample image to validate basic functionality."""
        if self.model is None or self.processor is None:
            logger.error("Model or processor not loaded")
            return False

        try:
            # Use a placeholder image if none provided
            if image_path is None or not os.path.exists(image_path):
                # Create a simple image with text for testing
                logger.info("Creating test image with sample text")
                from PIL import Image, ImageDraw, ImageFont

                # Create a white image
                img = Image.new("RGB", (800, 300), color="white")
                d = ImageDraw.Draw(img)

                # Try to use a default font
                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except Exception as _:
                    # Fall back to default font if arial not available
                    font = ImageFont.load_default()

                # Draw some HVAC-related text
                blueprint_text = "HVAC System Design\nSupply Temperature: 55°F\nReturn Temperature: 75°F\nAir Flow: 2000 CFM\nEquipment: VAV with reheat"
                d.text((50, 50), blueprint_text, fill="black", font=font)

                # Save temporary image
                temp_image_path = "temp_test_image.png"
                img.save(temp_image_path)
                image_path = temp_image_path
                logger.info("Created test image at %s", image_path)

            # Load and process the image
            logger.info("Processing image: %s", image_path)
            image = Image.open(image_path).convert("RGB")

            # Process image according to model type
            if self.model_type == "trocr":
                # Process for TrOCR
                pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(
                    self.device
                )

                # Generate text
                with torch.no_grad():
                    generated_ids = self.model.generate(pixel_values, max_length=50)

                # Decode the generated ids
                generated_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

                logger.info("TrOCR recognized text: %s", generated_text)

            elif self.model_type == "mllama":
                # Process for Llama Vision
                prompt = "What text do you see in this image?"
                inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
                    self.device
                )

                # Generate text
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                    )

                # Decode the output
                generated_text = self.processor.batch_decode(output, skip_special_tokens=True)[0]

                logger.info("Llama Vision response: %s", generated_text)

            # Clean up
            if image_path == "temp_test_image.png" and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    logger.info("Removed temporary test image")
                except Exception as e:
                    logger.error("Error validating sample image: %s", e)

            return len(generated_text) > 10  # Basic check if output is non-empty
        except Exception as e:
            logger.error("Error validating sample image: %s", e)
            return False

    def validate_blueprint_specific_capabilities(self, blueprint_path=None):
        """
        Validate blueprint-specific capabilities using test prompts.

        For now, uses test prompts since actual blueprints aren't available yet.
        """
        if self.model is None or self.processor is None:
            logger.error("Model or processor not loaded")
            return False

        try:
            # Create a test image with blueprint-like content if no blueprint is provided
            if blueprint_path is None or not os.path.exists(blueprint_path):
                logger.info("Creating test blueprint image")

                # Create a white image
                img = Image.new("RGB", (1200, 800), color="white")
                d = ImageDraw.Draw(img)

                # Try to use a default font
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                    title_font = ImageFont.truetype("arial.ttf", 24)
                except Exception as e:
                    # Fall back to default font if arial not available
                    font = ImageFont.load_default()
                    title_font = ImageFont.load_default()
                    logger.error("Error validating sample image: %s", e)

                # Draw blueprint-like content
                d.text((50, 30), "HVAC SYSTEM LAYOUT - FLOOR 1", fill="black", font=title_font)

                # Draw some rectangles to represent rooms
                d.rectangle([(100, 100), (400, 300)], outline="black")
                d.text((200, 150), "ROOM 101", fill="black", font=font)
                d.text((110, 180), "Supply: VAV-1", fill="black", font=font)
                d.text((110, 200), "Return: RTU-1", fill="black", font=font)
                d.text((110, 220), "CFM: 1000", fill="black", font=font)

                d.rectangle([(500, 100), (800, 300)], outline="black")
                d.text((600, 150), "ROOM 102", fill="black", font=font)
                d.text((510, 180), "Supply: VAV-2", fill="black", font=font)
                d.text((510, 200), "Return: RTU-1", fill="black", font=font)
                d.text((510, 220), "CFM: 1200", fill="black", font=font)

                # Draw some lines to represent ducts
                d.line([(250, 100), (250, 50), (650, 50), (650, 100)], fill="blue", width=2)

                # Add some labels for equipment
                d.rectangle([(900, 150), (1100, 250)], outline="red")
                d.text((950, 190), "AHU-1", fill="red", font=font)

                # Add some technical specifications
                specs = [
                    "GENERAL NOTES:",
                    "1. All ductwork shall comply with SMACNA standards",
                    "2. All VAV boxes to include reheat coils",
                    "3. Supply air temperature: 55°F",
                    "4. Return air temperature: 75°F",
                    "5. Cooling capacity: 20 tons",
                ]

                for i, spec in enumerate(specs):
                    d.text((100, 400 + i * 25), spec, fill="black", font=font)

                # Save temporary image
                blueprint_path = "temp_test_blueprint.png"
                img.save(blueprint_path)
                logger.info("Created test blueprint at %s", blueprint_path)

            # Load the blueprint image
            blueprint = Image.open(blueprint_path).convert("RGB")

            # Define HVAC-specific prompts to test blueprint reading capabilities
            blueprint_prompts = [
                "What rooms are shown in this blueprint?",
                "What is the air flow (CFM) for each room?",
                "Describe the HVAC equipment shown in this blueprint.",
                "What are the supply and return air temperatures?",
                "List all the VAV boxes and their associated rooms.",
            ]

            results = []

            for prompt in blueprint_prompts:
                logger.info("Testing blueprint reading with prompt: %s", prompt)

                if self.model_type == "trocr":
                    # TrOCR doesn't support prompting, so we just extract text
                    pixel_values = self.processor(blueprint, return_tensors="pt").pixel_values.to(
                        self.device
                    )

                    with torch.no_grad():
                        generated_ids = self.model.generate(pixel_values, max_length=200)

                    extracted_text = self.processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0]

                    results.append(
                        {
                            "prompt": prompt,
                            "response": "TrOCR extracted text: " + extracted_text,
                            "success": len(extracted_text)
                            > 20,  # Basic check if output is meaningful
                        }
                    )

                elif self.model_type == "mllama":
                    # Process for Llama Vision
                    inputs = self.processor(text=prompt, images=blueprint, return_tensors="pt").to(
                        self.device
                    )

                    with torch.no_grad():
                        output = self.model.generate(
                            **inputs,
                            max_new_tokens=200,
                            do_sample=True,
                            temperature=0.7,
                        )

                    response = self.processor.batch_decode(output, skip_special_tokens=True)[0]

                    # Check if response contains HVAC-specific terms
                    hvac_terms = ["room", "vav", "cfm", "temperature", "supply", "return", "ahu"]
                    hvac_term_matches = sum(
                        1 for term in hvac_terms if term.lower() in response.lower()
                    )

                    results.append(
                        {
                            "prompt": prompt,
                            "response": response,
                            "success": hvac_term_matches
                            >= 2,  # At least 2 HVAC terms should be in the response
                        }
                    )

                # Log a preview of the response
                logger.info("Response (truncated): %s", results[-1]["response"][:100] + "...")

            # Clean up
            if blueprint_path == "temp_test_blueprint.png" and os.path.exists(blueprint_path):
                try:
                    os.remove(blueprint_path)
                    logger.info("Removed temporary test blueprint")
                except Exception as e:
                    logger.error("Error validating sample image: %s", e)

            # Calculate success rate
            success_rate = sum(1 for result in results if result["success"]) / len(results)
            logger.info("Blueprint reading success rate: %s", f"{success_rate:.2f}")

            return success_rate > 0.5  # At least 50% success rate
        except Exception as e:
            logger.error("Error validating blueprint capabilities: %s", e)
            return False

    def run_full_validation(self, blueprint_path=None):
        """Run all validation tests and return a comprehensive report."""
        validation_results = {
            "model_loading": False,
            "model_architecture": False,
            "tensor_properties": False,
            "sample_image_test": False,
            "blueprint_capabilities": False,
        }

        try:
            # Step 1: Load the model
            validation_results["model_loading"] = self.load_model()
            if not validation_results["model_loading"]:
                logger.error("Model loading failed! Skipping remaining tests.")
                return validation_results

            # Step 2: Validate model architecture
            validation_results["model_architecture"] = self.validate_model_architecture()

            # Step 3: Validate tensor properties
            validation_results["tensor_properties"] = self.validate_tensor_properties()

            # Step 4: Test with a sample image
            validation_results["sample_image_test"] = self.validate_sample_image()

            # Step 5: Test blueprint-specific capabilities
            validation_results["blueprint_capabilities"] = (
                self.validate_blueprint_specific_capabilities(blueprint_path)
            )

            # Generate summary report
            logger.info("\n" + "=" * 50)
            logger.info("VISION MODEL VALIDATION SUMMARY")
            logger.info("=" * 50)

            all_tests_passed = True
            for test_name, result in validation_results.items():
                status = "PASSED" if result else "FAILED"
                logger.info("%s: %s", test_name.replace("_", " ").title(), status)
                all_tests_passed = all_tests_passed and result

            logger.info("-" * 50)
            logger.info("Overall validation: %s", "PASSED" if all_tests_passed else "FAILED")
            logger.info("=" * 50)

            return validation_results
        except Exception as e:
            logger.error("Error running full validation: %s", e)
            return validation_results
        finally:
            # Clean up resources
            if hasattr(self, "model") and self.model is not None:
                del self.model
            if hasattr(self, "processor") and self.processor is not None:
                del self.processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    # Accept model_id from command line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate Vision/OCR models for processing blueprints"
    )
    parser.add_argument(
        "--model_id", type=str, required=True, help="HuggingFace model ID to validate"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["trocr", "mllama"],
        default=None,
        help="Model type (will auto-detect if not specified)",
    )
    parser.add_argument(
        "--blueprint",
        type=str,
        default=None,
        help="Path to blueprint image for testing (will create a simple test image if not provided)",
    )

    args = parser.parse_args()

    # Use the provided model_id
    logger.info("Validating model: %s", args.model_id)
    validator = VisionModelValidator(model_id=args.model_id, model_type=args.model_type)
    validation_results = validator.run_full_validation(blueprint_path=args.blueprint)

    # Save results to file
    with open("vision_model_validation_results.json", "w", encoding="utf-8") as f:
        json.dump(validation_results, f, indent=4)
