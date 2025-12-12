# scripts/evaluate_final.py
"""
Final, robust evaluation script for Java Method Naming (Assignment 1).
Loads the model using explicit PEFT/Transformers methods to ensure
compatibility after manual saving/resizing, ignoring corrupted base model IDs.
"""

import torch
import json
import argparse
import sys
import os
from tqdm import tqdm

# Import necessary libraries for model loading
try:
    from peft import PeftModelForCausalLM
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    from unsloth import FastLanguageModel # Only for patching
except ImportError:
    print("Error: Please ensure 'transformers', 'peft', and 'unsloth' are installed.")
    sys.exit(1)


class MethodNamingEvaluator:
    
    # FIM tokens (used for tokenizer initialization and decoding)
    FIM_TOKENS = ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|endoftext|>"]
    END_OF_TEXT = "<|endoftext|>"
    BASE_MODEL_ID = "unsloth/Qwen2.5-Coder-0.5B" # 明确的基础模型ID

    def __init__(self, model_dir, max_seq_length=1024):
        """
        Initialize evaluation engine by loading the model and tokenizer.
        """
        self.model_dir = model_dir
        self.MAX_SEQ_LENGTH = max_seq_length
        
        print(f"Loading model and tokenizer from {model_dir}...")

        # 1. Load Tokenizer and add FIM tokens
        # Tokenizer is loaded from the final saved directory
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        # 显式添加FIM token以确保词汇表大小一致，即使文件已包含
        self.tokenizer.add_special_tokens({"additional_special_tokens": self.FIM_TOKENS})
        
        # 2. Load Base Model Configuration (从Hub加载基础模型的配置，以确保架构正确)
        config = AutoConfig.from_pretrained(self.BASE_MODEL_ID, trust_remote_code=True)
        
        # 3. Load Base Model from Hub, ignoring the broken local config reference
        base_model = AutoModelForCausalLM.from_pretrained(
            self.BASE_MODEL_ID,
            config=config,
            torch_dtype=torch.float16, 
            device_map="cpu", # 加载到CPU
            trust_remote_code=True,
        )
        
        # 4. Resize Embeddings to match the trained size
        # 这一步必须在加载 LoRA 之前完成
        base_model.resize_token_embeddings(len(self.tokenizer))
        
        # 5. Load LoRA Adapter onto the correctly sized Base Model
        # model_dir 包含 adapter_config.json 和 adapter_model.safetensors
        self.model = PeftModelForCausalLM.from_pretrained(
            base_model,
            self.model_dir,
            torch_dtype=torch.float16,
            is_trainable=False,
            # 最终的保障：强制忽略由于不规范保存导致的 Embedding 尺寸 mismatch
            ignore_mismatched_sizes=True 
        )
        
        # 6. Move model to GPU and set to eval mode
        if torch.cuda.is_available():
            # 启用 Unsloth 补丁的逻辑 (在 PeftModel 上应用)
            self.model.merge_and_unload() # 这是一个可选步骤，但可以简化推理，或直接使用 cuda()
            self.model = self.model.cuda()
            
        self.model.eval()
        print(f"✅ Model successfully loaded. Vocab size: {len(self.tokenizer)}.")
        print("Note: Model was explicitly loaded from base ID, then LoRA applied.")


    # ... (load_test_data, predict_method_name, evaluate, main 等函数保持不变) ...

    def load_test_data(self, test_path):
        """
        Loads test data from the JSONL file containing FIM-formatted data.
        The data is expected to contain a 'text' field which is the prompt 
        (e.g., fim_test.jsonl).
        """
        # (保持不变)
        test_data = []
        try:
            with open(test_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    full_text = data.get('text', '')
                    
                    if self.FIM_TOKENS[2] in full_text and self.END_OF_TEXT in full_text:
                        prompt, target_suffix = full_text.split(self.FIM_TOKENS[2], 1)
                        prompt += self.FIM_TOKENS[2]
                        true_name = target_suffix.split(self.END_OF_TEXT)[0].strip()

                        test_data.append({
                            'prompt': prompt,
                            'true_name': true_name
                        })

            print(f"Loaded {len(test_data)} FIM test samples successfully.")
        except FileNotFoundError:
            print(f"Error: Test data file not found at {test_path}")
            sys.exit(1)

        return test_data

    def predict_method_name(self, prompt):
        """Predict method name for a given FIM prompt."""
        # (保持不变)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.MAX_SEQ_LENGTH)

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][len(inputs['input_ids'][0]):]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)

        predicted_name = generated_text.split(self.END_OF_TEXT)[0].strip()

        return predicted_name.split('<')[0].strip()


    def evaluate(self, test_data, max_samples=None):
        # (保持不变)
        if max_samples is not None:
            test_data = test_data[:max_samples]

        if not test_data:
            return 0.0, 0.0, []

        correct = 0
        results = []
        total_samples = len(test_data)

        print(f"Starting evaluation on {total_samples} samples...")
        
        for i, item in enumerate(tqdm(test_data, total=total_samples, desc="Evaluation Progress")):
            true_name = item['true_name']
            predicted_name = self.predict_method_name(item['prompt'])

            match = predicted_name == true_name
            if match:
                correct += 1

            results.append({
                "index": i,
                "true_name": true_name,
                "predicted_name": predicted_name,
                "exact_match": match 
            })
            
            if (i + 1) % 500 == 0 or (i + 1) == total_samples:
                current_accuracy = correct / (i + 1) * 100
                tqdm.write(f"Processed {i + 1}/{total_samples} samples. Current Accuracy: {current_accuracy:.2f}%")

        accuracy = correct / total_samples * 100

        return accuracy, 0.0, results 


def main():
    # (保持不变)
    parser = argparse.ArgumentParser(description='Evaluate Java method naming model')
    parser.add_argument('--model-dir', required=True, help='Path to trained model (e.g., qwen_method_namer_model)')
    parser.add_argument('--test-data', required=True, help='Path to FIM test data JSONL (e.g., data/methods/fim_test.jsonl)') 
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to evaluate (default: None for all samples)')
    parser.add_argument('--output', default='evaluation_results.json', help='Output file path')

    args = parser.parse_args()

    evaluator = MethodNamingEvaluator(args.model_dir)
    test_data = evaluator.load_test_data(args.test_data)

    accuracy, _, results = evaluator.evaluate(test_data, args.max_samples)

    output_data = {
        "accuracy": accuracy,
        "evaluated_samples": len(results),
        "results": results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("
" + "="*50)
    print("✅ Evaluation completed!")
    print(f"Final Exact Match Accuracy: {accuracy:.2f}%")
    print(f"Total Samples Evaluated: {len(results)}")
    print(f"Results saved to: {args.output}")
    print("="*50)

if __name__ == "__main__":
    main()

# --- End of evaluate_final.py ---
