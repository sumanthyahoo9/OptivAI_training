"""
For our purposes, we're starting out with LlaMa 3.1 8B parameter model
This gives a good balance between size and capabilities for a large language model
"""

import gc
import os
import json
import logging
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
import matplotlib.pyplot as plt
from tqdm import tqdm

# If the model isn't available locally
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LlamaModelValidator:
    """
    Validate a Llama language model
    """

    def __init__(self, model_id="meta-llama/Llama-3.1-8B", use_cache=True):
        """
        Initialize the LlamaModelValidator with a specific model ID.

        Args:
            model_id: HuggingFace model ID
            use_cache: Whether to use cached models
        """
        self.model_id = model_id
        self.use_cache = use_cache
        self.model = None
        self.tokenizer = None
        self.safetensor_files = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def load_model(self):
        """Load the model and tokenizer from HuggingFace."""
        try:
            logger.info(f"Loading tokenizer from {self.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            logger.info(f"Loading model from {self.model_id}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                use_cache=self.use_cache,
            )
            logger.info("Model and tokenizer loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def load_safetensors(self, safetensor_dir=None):
        """
        Load and examine safetensor files.

        Args:
            safetensor_dir: Directory containing safetensor files
        """
        if safetensor_dir is None:
            # Try to find the safetensor directory in the huggingface cache
            try:
                safetensor_dir = snapshot_download(self.model_id)
            except Exception as e:
                logger.error(f"Error finding safetensor directory: {e}")
                return False

        logger.info(f"Examining safetensor files in {safetensor_dir}")

        # Find all safetensor files
        self.safetensor_files = []
        for root, _, files in os.walk(safetensor_dir):
            for file in files:
                if file.endswith(".safetensors"):
                    self.safetensor_files.append(os.path.join(root, file))

        logger.info(f"Found {len(self.safetensor_files)} safetensor files")

        # Load and examine a few safetensor files for validation
        if self.safetensor_files:
            tensor_stats = {}
            for file_path in self.safetensor_files[:5]:  # Examine just a few files
                try:
                    tensors = load_file(file_path)
                    tensor_stats[os.path.basename(file_path)] = {
                        "num_tensors": len(tensors),
                        "keys": list(tensors.keys())[:5],  # First 5 keys
                        "shapes": {k: tensors[k].shape for k in list(tensors.keys())[:5]},
                    }
                except Exception as e:
                    logger.error(f"Error examining safetensor file {file_path}: {e}")

            logger.info(f"Examined tensor statistics: {tensor_stats}")
            return True
        else:
            logger.warning("No safetensor files found")
            return False

    def validate_model_architecture(self):
        """Validate the model architecture."""
        if self.model is None:
            logger.error("Model not loaded")
            return False

        try:
            # Get model configuration
            config = self.model.config

            # Basic architecture checks for LLaMA models
            expected_config_keys = [
                "vocab_size",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
            ]

            # Verify the model has the expected configuration keys
            for key in expected_config_keys:
                if not hasattr(config, key):
                    logger.warning(f"Model configuration missing expected key: {key}")

            # Log key configuration parameters
            logger.info("Model architecture validation:")
            logger.info(f"  Model type: {config.model_type}")
            logger.info(f"  Vocab size: {config.vocab_size}")
            logger.info(f"  Hidden size: {config.hidden_size}")
            logger.info(f"  Number of layers: {config.num_hidden_layers}")
            logger.info(f"  Number of attention heads: {config.num_attention_heads}")

            # Validate model size is approximately 8B parameters
            model_parameters = sum(p.numel() for p in self.model.parameters())
            expected_parameters = 8e9  # 8 billion
            tolerance = 0.2  # 20% tolerance

            logger.info(f"  Total parameters: {model_parameters:,}")

            if abs(model_parameters - expected_parameters) / expected_parameters > tolerance:
                logger.warning(
                    f"Model size ({model_parameters:,}) differ significantly from expected size ({expected_parameters:,.0f})"
                )
            else:
                logger.info("  Model size matches expected 8B parameters")

            return True
        except Exception as e:
            logger.error(f"Error validating model architecture: {e}")
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
                    logger.warning(f"NaN values detected in {name}")

                if torch.isinf(param).any():
                    tensor_stats["has_inf"] = True
                    logger.warning(f"Inf values detected in {name}")

                # Calculate weight norm
                norm = torch.norm(param).item()
                tensor_stats["weight_norms"].append(norm)
                tensor_stats["layer_names"].append(name)

            # Check for consistent dtype usage
            logger.info(f"Data type distribution: {tensor_stats['dtypes']}")
            if (
                len(tensor_stats["dtypes"]) > 2
            ):  # Allowing for mixed precision (e.g., bf16 and float32)
                logger.warning(f"Model uses multiple data types: {tensor_stats['dtypes']}")

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
            logger.error(f"Error validating tensor properties: {e}")
            return False

    def validate_text_generation(self, prompt_texts=None):
        """
        Validate the model's text generation capabilities.

        Args:
            prompt_texts: List of prompt texts for testing
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model or tokenizer not loaded")
            return False

        if prompt_texts is None:
            # Default prompts related to HVAC control systems
            prompt_texts = [
                "Explain how to optimize HVAC energy consumption in a building.",
                "What is the relationship between indoor temperature, humidity, and energy consumption?",
                "List three strategies for reducing HVAC power demand during peak hours.",
            ]

        try:
            logger.info("Testing text generation capabilities")

            generation_success = True
            for i, prompt in enumerate(prompt_texts):
                logger.info(f"Testing generation for prompt {i + 1}: '{prompt[:50]}...'")

                # Tokenize the prompt
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                # Generate text
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                # Decode the output
                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

                # Check if the generated text is reasonable
                if len(generated_text) <= len(prompt) + 10:
                    logger.warning(f"Generated text is too short: '{generated_text}'")
                    generation_success = False
                else:
                    logger.info(f"Generated response (truncated): '{generated_text[:100]}...'")

                # Avoid GPU memory buildup
                del inputs, output
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            return generation_success
        except Exception as e:
            logger.error(f"Error validating text generation: {e}")
            return False

    def validate_hvac_specific_outputs(self):
        """
        Validate the model's outputs for HVAC-specific tasks based on the provided documents.
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model or tokenizer not loaded")
            return False

        try:
            # HVAC-specific prompts based on the provided documents
            hvac_prompts = [
                # From EnergyPlus simulation
                "Given an outdoor temperature of 25°C and indoor setpoint of 23°C, "
                "explain a strategy to minimize HVAC electricity demand.",
                # From paper concepts
                "How would you implement a Deep Q-Network (DQN) for HVAC control optimization?",
                # Based on RL concepts in the documents
                "Compare SAC, TD3, and PPO algorithms for HVAC control in terms of sample efficiency and performance.",
                # Domain-specific validation
                "What are the key parameters to monitor when optimizing a 5ZoneAutoDXVAV HVAC system?",
            ]

            logger.info("Testing HVAC-specific outputs")

            # Expected keywords in responses for each prompt
            expected_keywords = [
                ["energy", "consumption", "temperature", "setpoint", "efficiency"],
                ["reinforcement", "learning", "state", "action", "reward", "q-value"],
                ["sample", "efficiency", "performance", "policy", "actor", "critic"],
                ["temperature", "humidity", "energy", "consumption", "setpoint", "zone"],
            ]

            validation_results = []

            for i, (prompt, keywords) in enumerate(zip(hvac_prompts, expected_keywords)):
                logger.info(f"Testing HVAC prompt {i + 1}: '{prompt[:50]}...'")

                # Tokenize the prompt
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                # Generate text
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=200,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                # Decode the output
                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

                # Check for expected keywords in the response
                keyword_matches = sum(1 for kw in keywords if kw.lower() in generated_text.lower())
                keyword_coverage = keyword_matches / len(keywords)

                logger.info(
                    f"Keyword coverage: {keyword_coverage:.2f} ({keyword_matches}/{len(keywords)})"
                )

                validation_results.append(
                    {
                        "prompt": prompt,
                        "keyword_coverage": keyword_coverage,
                        "response_length": len(generated_text) - len(prompt),
                    }
                )

                # Avoid GPU memory buildup
                del inputs, output
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Compute overall validation score
            avg_keyword_coverage = sum(r["keyword_coverage"] for r in validation_results) / len(
                validation_results
            )
            avg_response_length = sum(r["response_length"] for r in validation_results) / len(
                validation_results
            )

            logger.info(f"Average keyword coverage: {avg_keyword_coverage:.2f}")
            logger.info(f"Average response length: {avg_response_length:.0f} characters")

            return avg_keyword_coverage >= 0.6  # At least 60% keyword coverage for passing
        except Exception as e:
            logger.error(f"Error validating HVAC-specific outputs: {e}")
            return False

    def test_numerical_reasoning(self):
        """Test the model's numerical reasoning capabilities for HVAC calculations."""
        if self.model is None or self.tokenizer is None:
            logger.error("Model or tokenizer not loaded")
            return False

        try:
            # Simple HVAC-related calculations
            calculation_prompts = [
                "If an HVAC system consumes 5 kW for 10 hours, how much energy does it consume in kWh?",
                "If the outdoor temperature is 30°C and the cooling setpoint is 24°C with a coefficient of performance of 3,"
                "calculate the energy needed to cool 100 cubic meters of air.",
                "A building has an average power demand of 150 kW. "
                "If the demand charge is $15.65/kW, what is the monthly demand charge?",
            ]

            # Expected answers (approximately)
            expected_answers = [
                "50 kWh",  # 5 kW * 10 h = 50 kWh
                "calculation involving temperature difference of 6°C",  # Looking for methodology
                "$2,347.50",  # 150 kW * $15.65/kW = $2,347.50
            ]

            logger.info("Testing numerical reasoning for HVAC calculations")

            numerical_reasoning_score = 0

            for i, (prompt, expected) in enumerate(zip(calculation_prompts, expected_answers)):
                logger.info(f"Testing calculation {i + 1}: '{prompt}'")

                # Tokenize the prompt
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                # Generate text
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.1,  # Low temperature for more deterministic outputs
                        do_sample=False,  # Greedy decoding for calculations
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                # Decode the output
                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

                # Very basic check if the expected answer is in the response
                if expected.lower() in generated_text.lower():
                    numerical_reasoning_score += 1
                    logger.info(f"Calculation correct: Expected '{expected}' found in response")
                else:
                    logger.warning(
                        f"Calculation potentially incorrect: Expected '{expected}' not found"
                    )
                    logger.info(f"Response (truncated): '{generated_text[len(prompt):100]}...'")

                # Avoid GPU memory buildup
                del inputs, output
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            success_rate = numerical_reasoning_score / len(calculation_prompts)
            logger.info(f"Numerical reasoning success rate: {success_rate:.2f}")

            return success_rate >= 0.5  # At least 50% correct for passing
        except Exception as e:
            logger.error(f"Error testing numerical reasoning: {e}")
            return False

    def run_full_validation(self):
        """Run all validation tests and return a comprehensive report."""
        validation_results = {
            "model_loading": False,
            "safetensor_inspection": False,
            "model_architecture": False,
            "tensor_properties": False,
            "text_generation": False,
            "hvac_specific_validation": False,
            "numerical_reasoning": False,
        }

        try:
            # Step 1: Load the model
            validation_results["model_loading"] = self.load_model()
            if not validation_results["model_loading"]:
                logger.error("Model loading failed! Skipping remaining tests.")
                return validation_results

            # Step 2: Inspect safetensor files
            validation_results["safetensor_inspection"] = self.load_safetensors()

            # Step 3: Validate model architecture
            validation_results["model_architecture"] = self.validate_model_architecture()

            # Step 4: Validate tensor properties
            validation_results["tensor_properties"] = self.validate_tensor_properties()

            # Step 5: Validate text generation capabilities
            validation_results["text_generation"] = self.validate_text_generation()

            # Step 6: HVAC-specific validation
            validation_results["hvac_specific_validation"] = self.validate_hvac_specific_outputs()

            # Step 7: Test numerical reasoning
            validation_results["numerical_reasoning"] = self.test_numerical_reasoning()

            # Generate summary report
            logger.info("\n" + "=" * 50)
            logger.info("MODEL VALIDATION SUMMARY")
            logger.info("=" * 50)

            all_tests_passed = True
            for test_name, result in validation_results.items():
                status = "PASSED" if result else "FAILED"
                logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
                all_tests_passed = all_tests_passed and result

            logger.info("-" * 50)
            logger.info(f"Overall validation: {'PASSED' if all_tests_passed else 'FAILED'}")
            logger.info("=" * 50)

            return validation_results
        except Exception as e:
            logger.error(f"Error running full validation: {e}")
            return validation_results
        finally:
            # Clean up resources
            if hasattr(self, "model") and self.model is not None:
                del self.model
            if hasattr(self, "tokenizer") and self.tokenizer is not None:
                del self.tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description="Validate LLM model prior to training")
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="HuggingFace model ID to validate",
    )
    args = parser.parse_args()

    # Use the provided model_id
    logger.info(f"Validating model: {args.model_id}")
    validator = LlamaModelValidator(model_id=args.model_id)
    validation_results = validator.run_full_validation()

    # Save results to file
    with open("model_validation_results.json", "w", encoding="utf-8") as f:
        json.dump(validation_results, f, indent=4)
