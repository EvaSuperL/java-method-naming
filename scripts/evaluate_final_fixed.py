# scripts/evaluate_final_fixed.py
"""
Final evaluation script – fully compatible with model saved via trainer.save_model()
"""

import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


class MethodNamingEvaluator:

    FIM_MIDDLE = "<|fim_middle|>"
    END_OF_TEXT = "<|endoftext|>"

    def __init__(self, model_dir, max_seq_length=1024):
        print(f"
🚀 Loading tokenizer from: {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        print(f"🚀 Loading FULL merged model from: {model_dir}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        self.model.eval()
        self.max_seq_length = max_seq_length

        print("✅ Model fully loaded and ready for evaluation!")

    # ---------------------------------------------------------
    def load_test_data(self, test_path):
        data = []

        print(f"
📥 Loading FIM test data: {test_path}")

        with open(test_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                full_text = obj["text"]

                if self.FIM_MIDDLE in full_text:
                    prompt, suffix = full_text.split(self.FIM_MIDDLE, 1)
                    prompt += self.FIM_MIDDLE
                    true_name = suffix.split(self.END_OF_TEXT)[0].strip()

                    data.append({
                        "prompt": prompt,
                        "true_name": true_name
                    })

        print(f"📊 Loaded {len(data)} test samples")
        return data

    # ---------------------------------------------------------
    def predict_method_name(self, prompt):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length
        )

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

        generated_tokens = outputs[0][len(inputs["input_ids"][0]):]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)

        pred = text.split(self.END_OF_TEXT)[0].strip()
        pred = pred.split("<")[0].strip()

        return pred

    # ---------------------------------------------------------
    def evaluate(self, test_data):
        correct = 0
        results = []

        print("
🏁 Starting evaluation...
")

        for i, item in enumerate(tqdm(test_data, desc="Evaluating")):
            true_name = item["true_name"]
            pred = self.predict_method_name(item["prompt"])

            exact = (pred == true_name)
            if exact:
                correct += 1

            results.append({
                "index": i,
                "true_name": true_name,
                "predicted_name": pred,
                "exact_match": exact
            })

        accuracy = correct / len(test_data) * 100
        print(f"
🎉 Final Exact Match Accuracy: {accuracy:.2f}%")

        return accuracy, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--output", default="evaluation_results.json")
    args = parser.parse_args()

    evaluator = MethodNamingEvaluator(args.model_dir)
    test_data = evaluator.load_test_data(args.test_data)
    accuracy, results = evaluator.evaluate(test_data)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "samples": len(results),
            "results": results
        }, f, indent=2)

    print(f"📄 Results saved to: {args.output}")


if __name__ == "__main__":
    main()

