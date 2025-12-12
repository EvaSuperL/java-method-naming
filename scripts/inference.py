# scripts/inference.py
"""
Inference script for Java method naming model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
import os

class MethodNamingInference:
    """
    Inference engine for method naming using FIM format
    """

    def __init__(self, model_dir):
        """
        Initialize inference engine

        Args:
            model_dir: Directory containing the trained model
        """
        self.model_dir = model_dir

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)

        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model.eval()

        # FIM tokens
        self.FIM_PREFIX = "<|fim_prefix|>"
        self.FIM_SUFFIX = "<|fim_suffix|>"
        self.FIM_MIDDLE = "<|fim_middle|>"
        self.END_OF_TEXT = "<|endoftext|>"

    def _find_method_name_position(self, method_body):
        """
        Find where to place <MASK> in the method body
        """
        lines = method_body.strip().split('\n')
        if not lines:
            return None, None

        # Find signature line
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith('//') or line_stripped.startswith('/*'):
                continue
            if '(' in line and ')' in line:
                # Try to find method name
                before_paren = line.split('(')[0]
                words = before_paren.strip().split()
                if len(words) > 1:
                    # Assume last word before '(' is method name
                    potential_name = words[-1]
                    start_idx = line.rfind(potential_name)
                    if start_idx != -1:
                        return i, start_idx

        return None, None

    def create_fim_input(self, method_body):
        """
        Create FIM format input from method body
        """
        # Find where to mask
        line_idx, char_idx = self._find_method_name_position(method_body)

        if line_idx is None:
            return None

        lines = method_body.strip().split('\n')
        signature_line = lines[line_idx]

        # Create masked line
        masked_line = signature_line[:char_idx] + "<MASK>" + signature_line[char_idx + len("<MASK>"):]
        lines[line_idx] = masked_line
        masked_body = '\n'.join(lines)

        # Create FIM format
        mask_pos = masked_body.find("<MASK>")
        prefix = masked_body[:mask_pos]
        suffix = masked_body[mask_pos + 6:]  # Length of "<MASK>"

        fim_input = f"{self.FIM_PREFIX}{prefix}{self.FIM_SUFFIX}{suffix}{self.FIM_MIDDLE}"

        return fim_input

    def predict_method_name(self, method_body):
        """
        Predict method name for a given method body
        """
        # Create FIM input
        fim_input = self.create_fim_input(method_body)
        if not fim_input:
            return ""

        # Tokenize
        inputs = self.tokenizer(fim_input, return_tensors="pt")

        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract method name between <|fim_middle|> and <|endoftext|>
        start_marker = self.FIM_MIDDLE
        end_marker = self.END_OF_TEXT

        start_idx = generated.find(start_marker)
        if start_idx != -1:
            start_idx += len(start_marker)
            end_idx = generated.find(end_marker, start_idx)
            if end_idx != -1:
                predicted = generated[start_idx:end_idx].strip()
                # Clean up
                predicted = predicted.split('<')[0].strip()
                return predicted

        return ""

    def evaluate(self, test_data, max_samples=None):
        """
        Evaluate model on test data

        Args:
            test_data: List of dicts with 'name' and 'body' keys
            max_samples: Maximum number of samples to evaluate

        Returns:
            accuracy: Percentage of correct predictions
            results: List of prediction results
        """
        if max_samples:
            test_data = test_data[:max_samples]

        correct = 0
        results = []

        for i, item in enumerate(test_data):
            true_name = item['name']
            predicted_name = self.predict_method_name(item['body'])

            match = predicted_name.lower() == true_name.lower()
            if match:
                correct += 1

            results.append({
                "index": i,
                "true_name": true_name,
                "predicted_name": predicted_name,
                "correct": match
            })

            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(test_data)} samples...")

        accuracy = correct / len(test_data) * 100 if test_data else 0

        return accuracy, results

def load_test_data(test_path):
    """Load test data from JSONL file"""
    test_data = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Convert from FIM format back to original format
            if 'name' in data and 'body' in data:
                test_data.append(data)
    return test_data

def main():
    """Main function for standalone execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate method naming model')
    parser.add_argument('--model-dir', required=True, help='Path to trained model')
    parser.add_argument('--test-data', required=True, help='Path to test data JSONL')
    parser.add_argument('--max-samples', type=int, default=100, help='Max samples to evaluate')
    parser.add_argument('--output', default='evaluation_results.json', help='Output file path')

    args = parser.parse_args()

    # Load test data
    print(f"Loading test data from {args.test_data}...")
    test_data = load_test_data(args.test_data)
    print(f"Loaded {len(test_data)} test samples")

    # Initialize inference engine
    print(f"Loading model from {args.model_dir}...")
    inference = MethodNamingInference(args.model_dir)

    # Evaluate
    print(f"Evaluating on {min(args.max_samples, len(test_data))} samples...")
    accuracy, results = inference.evaluate(test_data, args.max_samples)

    # Save results
    output_data = {
        "accuracy": accuracy,
        "evaluated_samples": len(results),
        "results": results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Evaluation completed!")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
