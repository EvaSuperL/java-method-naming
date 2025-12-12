# scripts/correct_evaluation_minimal_fix.py
"""
Minimal fix evaluation script - aligned with FIM training format
Keeps your original structure & keeps save_results()
"""

import json
import os
import sys
import torch
import transformers
import numpy
from datetime import datetime
from tqdm import tqdm

# Fix PyTorch 2.6 loading
torch.serialization.add_safe_globals([
    numpy._core.multiarray._reconstruct,
    transformers.training_args.TrainingArguments,
])


# ============================================================
#   Evaluator (minimal modifications, training-aligned FIM)
# ============================================================

class CorrectMethodNamingEvaluator:
    """Evaluator using correct FIM format (same as training)"""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.model_loaded = False

        print(f"Initializing evaluator, model directory: {model_dir}")

        try:
            self._load_model()
        except Exception as e:
            print(f"Model loading failed: {e}")
            sys.exit(1)


    # --------------------------------------------------------
    #               LOAD MODEL + TOKENIZER
    # --------------------------------------------------------
    def _load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel, PeftConfig

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        adapter_path = os.path.join(self.model_dir, "adapter_config.json")

        if os.path.exists(adapter_path):
            print("Detected PEFT adapter → loading base model + LoRA")

            config = PeftConfig.from_pretrained(self.model_dir)

            base = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None
            )

            base.resize_token_embeddings(len(self.tokenizer))

            self.model = PeftModel.from_pretrained(
                base,
                self.model_dir,
                torch_dtype=torch.float16
            )

        else:
            print("No LoRA adapter found → Loading full model")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None
            )

        self.model.eval()
        self.model_loaded = True
        print(f"Model loaded on: {self.model.device}")



    # --------------------------------------------------------
    #           TRAINING-ALIGNED FIM INPUT BUILDER
    # --------------------------------------------------------
    def _extract_method_name_from_body(self, body):
        """Find true method name (for masking)"""
        lines = body.strip().split("\n")

        for line in lines:
            line = line.strip()
            if "(" in line and ")" in line:
                parts = line.split("(")[0].split()
                if len(parts) >= 2:
                    return parts[-1]  # last token = method name
        return None


    def _create_fim_input(self, body):
        """
        EXACT SAME FORMAT AS TRAINING:

        prefix = part_before_method_name
        suffix = part_after_method_name

        <|fim_prefix|>prefix<|fim_suffix|>suffix<|fim_middle|>
        """
        name = self._extract_method_name_from_body(body)
        if not name:
            return None

        pos = body.find(name)
        if pos == -1:
            return None

        prefix = body[:pos]
        suffix = body[pos + len(name):]

        return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"


    # --------------------------------------------------------
    #                MODEL PREDICTION
    # --------------------------------------------------------
    def predict_method_name(self, body):
        """Predict method name using FIM masking"""

        fim_input = self._create_fim_input(body)
        if not fim_input:
            return ""

        inputs = self.tokenizer(
            fim_input,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        text = self.tokenizer.decode(output[0], skip_special_tokens=False)

        if "<|fim_middle|>" in text:
            part = text.split("<|fim_middle|>")[1]
            part = part.split("<|endoftext|>")[0]
            return part.strip()

        return ""


    # --------------------------------------------------------
    #            LOAD TEST DATA
    # --------------------------------------------------------
    def load_test_data(self, path):
        lst = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    if "name" in j and "body" in j:
                        lst.append(j)
                except:
                    pass

        print(f"Loaded {len(lst)} test samples")
        return lst


    # --------------------------------------------------------
    #            EVALUATION LOOP
    # --------------------------------------------------------
    def evaluate(self, test_data, max_samples=None):
        if max_samples:
            test_data = test_data[:max_samples]

        results = []
        correct = 0

        print(f"Evaluating {len(test_data)} samples...")

        for i, item in enumerate(tqdm(test_data)):
            truth = item["name"]
            pred = self.predict_method_name(item["body"])

            pred_clean = pred.split("(")[0].split()[0].strip() if pred else ""

            match = (pred_clean.lower() == truth.lower())
            if match:
                correct += 1

            results.append({
                "index": i,
                "true_name": truth,
                "predicted_name": pred_clean,
                "exact_match": match
            })

        acc = correct / len(test_data) * 100
        return acc, results


    # --------------------------------------------------------
    #             SAVE RESULTS (kept from your script)
    # --------------------------------------------------------
    def save_results(self, accuracy, results, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        detail_path = os.path.join(output_dir, "evaluation_detailed.json")
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump({
                "accuracy": accuracy,
                "total_samples": len(results),
                "exact_matches": sum(r["exact_match"] for r in results),
                "results": results[:500]
            }, f, indent=2)

        summary = (
            f"Java Method Naming Evaluation\n"
            f"Accuracy: {accuracy:.2f}%\n"
            f"Samples: {len(results)}\n"
            f"Correct: {sum(r['exact_match'] for r in results)}\n"
        )

        with open(os.path.join(output_dir, "evaluation_summary.txt"), "w") as f:
            f.write(summary)

        print("Results saved!")


# ============================================================
#               MAIN ENTRYPOINT
# ===========================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default="output/minimal_fix")

    args = parser.parse_args()

    evaluator = CorrectMethodNamingEvaluator(args.model_dir)
    test = evaluator.load_test_data(args.test_data)

    acc, results = evaluator.evaluate(test, args.max_samples)
    evaluator.save_results(acc, results, args.output_dir)

    print(f"\nFinal Accuracy: {acc:.2f}%")
    print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
