# scripts/inference.py
"""
Improved Inference script for Java method naming model
Fixed bitsandbytes compatibility issues
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

    def __init__(self, model_dir, use_4bit=False):
        """
        Initialize inference engine
        
        Args:
            model_dir: Directory containing the trained model
            use_4bit: Whether to use 4-bit quantization (may cause issues)
        """
        self.model_dir = model_dir
        self.use_4bit = use_4bit
        
        print(f"Initializing inference engine from: {model_dir}")
        
        try:
            # First try to load without 4-bit (safer)
            if use_4bit:
                # Try with 4-bit quantization
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # Load with float16 (more compatible)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"✅ Model loaded successfully")
            print(f"   Model device: {self.model.device}")
            print(f"   Model dtype: {self.model.dtype}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Trying alternative loading method...")
            
            # Alternative: Load base model and apply LoRA
            self._load_with_alternative_method()
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # FIM tokens
        self.FIM_PREFIX = "<|fim_prefix|>"
        self.FIM_SUFFIX = "<|fim_suffix|>"
        self.FIM_MIDDLE = "<|fim_middle|>"
        self.END_OF_TEXT = "<|endoftext|>"
    
    def _load_with_alternative_method(self):
        """Alternative method to load model (for compatibility)"""
        try:
            print("Loading base model and applying LoRA...")
            
            # Load base model
            from unsloth import FastLanguageModel
            
            base_model, tokenizer = FastLanguageModel.from_pretrained(
                model_name="unsloth/Qwen2.5-Coder-0.5B",
                max_seq_length=512,
                dtype=torch.float16,
                load_in_4bit=False,  # Disable 4-bit
            )
            
            # Load LoRA adapter
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base_model, self.model_dir)
            
            # Merge and unload for inference
            self.model = self.model.merge_and_unload()
            self.tokenizer = tokenizer
            
            print("✅ Model loaded via alternative method")
            
        except Exception as e:
            print(f"❌ Alternative method also failed: {e}")
            raise
    
    def _find_method_name_position(self, method_body):
        """
        Improved method to find where to place <MASK> in the method body
        Handles more Java signature patterns
        """
        lines = method_body.strip().split('\n')
        if not lines:
            return None, None
        
        # Common Java types for filtering
        java_types = {
            'void', 'int', 'String', 'boolean', 'float', 'double', 'long',
            'char', 'byte', 'short', 'List', 'Map', 'Set', 'ArrayList',
            'HashMap', 'HashSet', 'Object'
        }
        
        # Find signature line
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Skip empty lines and comments
            if not line_stripped:
                continue
            if line_stripped.startswith('//') or line_stripped.startswith('/*') or line_stripped.startswith('*'):
                continue
            
            # Check if this looks like a method signature
            if '(' in line and ')' in line:
                # Extract content before parentheses
                before_paren = line.split('(')[0].strip()
                words = before_paren.split()
                
                if len(words) >= 2:
                    # Look for method name (last word that's not a Java type)
                    for word in reversed(words):
                        clean_word = word.strip('*&<>[]')
                        if clean_word and clean_word not in java_types:
                            # Found potential method name
                            start_idx = line.rfind(clean_word)
                            if start_idx != -1:
                                return i, start_idx
        
        return None, None
    
    def create_fim_input(self, method_body):
        """
        Create FIM format input from method body
        """
        # Find where to mask
        result = self._find_method_name_position(method_body)
        if result is None:
            return None
        
        line_idx, char_idx = result
        
        lines = method_body.strip().split('\n')
        if line_idx >= len(lines):
            return None
        
        signature_line = lines[line_idx]
        
        # Find the actual word to mask
        before_paren = signature_line.split('(')[0]
        words = before_paren.split()
        java_types = {'void', 'int', 'String', 'boolean', 'float', 'double', 'long'}
        
        method_name = None
        for word in reversed(words):
            clean_word = word.strip('*&<>[]')
            if clean_word and clean_word not in java_types:
                method_name = clean_word
                break
        
        if method_name is None:
            return None
        
        # Create masked line
        masked_line = signature_line.replace(method_name, "<MASK>", 1)
        lines[line_idx] = masked_line
        masked_body = '\n'.join(lines)
        
        # Create FIM format
        mask_pos = masked_body.find("<MASK>")
        if mask_pos == -1:
            return None
        
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
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                fim_input,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Move to model device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=15,
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
                    # Clean up any remaining special tokens or whitespace
                    predicted = predicted.split('<')[0].strip()
                    return predicted
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
        
        return ""
    
    def evaluate(self, test_data, max_samples=100, verbose=True):
        """
        Evaluate model on test data
        
        Args:
            test_data: List of dicts with 'name' and 'body' keys
            max_samples: Maximum number of samples to evaluate
            verbose: Whether to print progress
        
        Returns:
            accuracy: Percentage of correct predictions
            results: List of prediction results
        """
        if not test_data:
            return 0.0, []
        
        if max_samples:
            test_data = test_data[:max_samples]
        
        correct = 0
        results = []
        
        if verbose:
            print(f"Evaluating on {len(test_data)} samples...")
        
        for i, item in enumerate(test_data):
            true_name = item.get('name', '')
            method_body = item.get('body', '')
            
            if not true_name or not method_body:
                results.append({
                    "index": i,
                    "true_name": true_name,
                    "predicted_name": "",
                    "correct": False,
                    "error": "Missing data"
                })
                continue
            
            try:
                predicted_name = self.predict_method_name(method_body)
                
                # Compare ignoring case and underscores
                true_clean = true_name.lower().replace('_', '')
                pred_clean = predicted_name.lower().replace('_', '')
                
                # Multiple matching criteria
                exact_match = (pred_clean == true_clean)
                contains_match = (true_clean in pred_clean) or (pred_clean in true_clean)
                
                match = exact_match or contains_match
                
                if match:
                    correct += 1
                
                results.append({
                    "index": i,
                    "true_name": true_name,
                    "predicted_name": predicted_name,
                    "correct": match,
                    "match_type": "exact" if exact_match else "partial" if contains_match else "none"
                })
                
            except Exception as e:
                results.append({
                    "index": i,
                    "true_name": true_name,
                    "predicted_name": "",
                    "correct": False,
                    "error": str(e)
                })
            
            # Print progress
            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(test_data)} samples...")
        
        accuracy = correct / len(test_data) * 100 if test_data else 0.0
        
        if verbose:
            print(f"✅ Evaluation completed:")
            print(f"   Samples: {len(test_data)}")
            print(f"   Correct: {correct}")
            print(f"   Accuracy: {accuracy:.2f}%")
        
        return accuracy, results

def load_test_data(test_path, max_samples=None):
    """Load test data from JSONL file"""
    test_data = []
    try:
        with open(test_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                try:
                    data = json.loads(line.strip())
                    if 'name' in data and 'body' in data:
                        test_data.append({
                            'name': data['name'],
                            'body': data['body']
                        })
                except json.JSONDecodeError:
                    continue
        
        print(f"Loaded {len(test_data)} test samples from {test_path}")
        return test_data
        
    except Exception as e:
        print(f"Error loading test data: {e}")
        return []

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Java method naming model')
    parser.add_argument('--model-dir', required=True, help='Path to trained model directory')
    parser.add_argument('--test-data', required=True, help='Path to test data JSONL file')
    parser.add_argument('--max-samples', type=int, default=100, help='Maximum samples to evaluate')
    parser.add_argument('--output', default='evaluation_results.json', help='Output JSON file')
    parser.add_argument('--no-4bit', action='store_true', help='Disable 4-bit quantization')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    print(f"Loading model from {args.model_dir}...")
    try:
        inference = MethodNamingInference(args.model_dir, use_4bit=not args.no_4bit)
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return
    
    # Load test data
    test_data = load_test_data(args.test_data, args.max_samples)
    if not test_data:
        print("No test data loaded. Exiting.")
        return
    
    # Evaluate
    accuracy, results = inference.evaluate(test_data, max_samples=args.max_samples)
    
    # Prepare output data
    output_data = {
        "model_path": args.model_dir,
        "test_data_path": args.test_data,
        "accuracy": accuracy,
        "evaluated_samples": len(results),
        "results": results[:20],  # Include first 20 results for analysis
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }
    
    # Save results
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to {args.output}")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    # Also print summary
    print(f"\n📊 SUMMARY:")
    print(f"  Model: {os.path.basename(args.model_dir)}")
    print(f"  Test samples: {len(test_data)}")
    print(f"  Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
