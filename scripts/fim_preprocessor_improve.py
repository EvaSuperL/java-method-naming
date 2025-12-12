# scripts/fim_preprocessor_improve.py
"""
FIM Format Preprocessor for Java Method Naming
Converts raw Java methods to FIM (Fill-in-the-Middle) format for training.
This script is robust as it uses the known method_name for slicing.
"""

import json
import re
import os
from tqdm import tqdm
import argparse
import sys

class FIMPreprocessor:
    """
    Preprocess Java methods into FIM format as required by the assignment
    """

    # FIM special tokens (Qwen format)
    FIM_PREFIX = "<|fim_prefix|>"
    FIM_SUFFIX = "<|fim_suffix|>"
    FIM_MIDDLE = "<|fim_middle|>"
    END_OF_TEXT = "<|endoftext|>"

    @staticmethod
    def create_fim_example(method_body, method_name):
        """
        Create FIM format training example using direct slicing.
        This method is robust because we use the known method_name for masking.
        Returns: (fim_input, fim_output) or (None, None) if failed
        """
        
        # 1. Find the position of the method name in the body.
        # Use rfind() to find the last occurrence, which is typically the method name in the signature.
        start_idx = method_body.rfind(method_name)
        
        if start_idx == -1:
            # The method name must be present in the body to be masked
            return None, None

        # 2. Split into prefix (before name) and suffix (after name)
        prefix = method_body[:start_idx]
        suffix = method_body[start_idx + len(method_name):]

        # 3. Create FIM format input (The method body with the name masked)
        fim_input = (
            f"{FIMPreprocessor.FIM_PREFIX}{prefix}"
            f"{FIMPreprocessor.FIM_SUFFIX}{suffix}"
            f"{FIMPreprocessor.FIM_MIDDLE}"
        )

        # 4. Create FIM format output (The target method name)
        fim_output = f"{method_name}{FIMPreprocessor.END_OF_TEXT}"

        return fim_input, fim_output

    @classmethod
    def process_jsonl_file(cls, input_path, output_path, max_samples=None):
        """
        Process a JSONL file from raw format (name, body) to FIM format (text)
        """
        if not os.path.exists(input_path):
             print(f"Error: Input file not found at {input_path}")
             sys.exit(1)

        print(f"Processing raw data from {input_path} to FIM format in {output_path}")

        processed_count = 0
        skipped_count = 0

        # Read the file twice: once for count, once for processing
        with open(input_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)

        if max_samples:
            total_lines = min(total_lines, max_samples)
        
        if total_lines == 0:
            print("Warning: Input file is empty.")
            return 0

        with open(input_path, 'r', encoding='utf-8') as infile,              open(output_path, 'w', encoding='utf-8') as outfile:

            for i, line in tqdm(enumerate(infile), total=total_lines, desc="FIM Preprocessing"):
                if max_samples and i >= max_samples:
                    break

                try:
                    data = json.loads(line.strip())
                    # Expecting raw format from github_miner.py: {"name": "...", "body": "..."}
                    method_body = data.get('body', '')
                    method_name = data.get('name', '')

                    if not method_body or not method_name:
                        skipped_count += 1
                        continue

                    # Create FIM example using the robust static method
                    fim_input, fim_output = cls.create_fim_example(method_body, method_name)

                    if fim_input and fim_output:
                        # Save as the combined 'text' field required by Unsloth/HuggingFace datasets
                        output_data = {
                            "text": fim_input + fim_output
                        }
                        outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                        processed_count += 1
                    else:
                        skipped_count += 1

                except Exception:
                    skipped_count += 1

        print(f"✅ FIM Preprocessing complete. Processed: {processed_count}, Skipped: {skipped_count}")
        return processed_count

def main():
    """Main function for standalone execution"""
    parser = argparse.ArgumentParser(description='Convert Java methods to FIM format')
    parser.add_argument('--input', required=True, help='Input JSONL file path (raw format: name, body)')
    parser.add_argument('--output', required=True, help='Output JSONL file path (FIM format: text)')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum number of samples to process')

    args = parser.parse_args()

    FIMPreprocessor.process_jsonl_file(args.input, args.output, args.max_samples)

if __name__ == "__main__":
    main()
