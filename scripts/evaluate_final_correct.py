# scripts/evaluate_final_correct.py

import torch
import json
from tqdm import tqdm
from unsloth import FastLanguageModel

class Evaluator:

    FIM_MIDDLE = "<|fim_middle|>"
    END = "<|endoftext|>"

    def __init__(self, model_dir):
        print(f"🔧 Loading model from {model_dir}")

        # Load model using Unsloth loader (this is CRITICAL)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )

        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

        print("✅ Model loaded successfully")
        print(f"Tokenizer vocab size: {len(self.tokenizer)}")

    def load_test_data(self, path):
        print(f"📄 Loading test set from {path}")
        data = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = obj["text"]

                if self.FIM_MIDDLE in text and self.END in text:
                    prompt, tail = text.split(self.FIM_MIDDLE, 1)
                    prompt = prompt + self.FIM_MIDDLE
                    true = tail.split(self.END)[0].strip()

                    data.append({"prompt": prompt, "true": true})

        print(f"Loaded {len(data)} test samples.")
        return data

    def predict(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=15,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen = self.tokenizer.decode(out[0], skip_special_tokens=False)
        gen = gen.split(self.FIM_MIDDLE)[-1]
        gen = gen.split(self.END)[0].strip()
        gen = gen.split("<")[0].strip()

        return gen

    def evaluate(self, dataset):
        correct = 0
        results = []

        print("🚀 Running evaluation...")
        for i, item in enumerate(tqdm(dataset)):
            pred = self.predict(item["prompt"])
            true = item["true"]

            ok = (pred == true)
            if ok:
                correct += 1

            results.append({
                "true": true,
                "predicted": pred,
                "exact_match": ok,
            })

        acc = correct / len(dataset) * 100
        return acc, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--output", default="evaluation_results.json")

    args = parser.parse_args()

    evaluator = Evaluator(args.model_dir)
    dataset = evaluator.load_test_data(args.test_data)

    acc, results = evaluator.evaluate(dataset)

    json.dump({
        "accuracy": acc,
        "total": len(results),
        "results_preview": results[:20],
    }, open(args.output, "w"), indent=2)

    print("===================================")
    print("🎉 Evaluation completed")
    print(f"Exact Match Accuracy = {acc:.2f}%")
    print(f"Saved results to: {args.output}")
    print("===================================")


# --- End of evaluate_final_correct.py ---
