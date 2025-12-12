# scripts/fim_preprocessor.py
"""
FIM Format Preprocessor for Java Method Naming
Converts raw Java methods to FIM (Fill-in-the-Middle) format for training
"""

import json
import re
import os
from tqdm import tqdm

class FIMPreprocessor:
    """
    Preprocess Java methods into FIM format as required by the assignment
    """

    # FIM special tokens (Qwen format)
    FIM_PREFIX = "<|fim_prefix|>"
    FIM_SUFFIX = "<|fim_suffix|>"
    FIM_MIDDLE = "<|fim_middle|>"
    END_OF_TEXT = "<|endoftext|>"

    # Java keywords and types for filtering
    JAVA_TYPES = {
        'void', 'int', 'String', 'boolean', 'float', 'double', 'long',
        'char', 'byte', 'short', 'List', 'Map', 'Set', 'ArrayList',
        'HashMap', 'HashSet', 'Object', 'Integer', 'Boolean', 'Float',
        'Double', 'Long', 'Character', 'Byte', 'Short'
    }

    @staticmethod
    def mask_method_signature(method_body):
        """
        Mask the method name in a Java method signature
        Example: "public static int sum(int a, int b)" -> "public static int <MASK>(int a, int b)"
        """
        lines = method_body.strip().split('\n')
        if not lines:
            return method_body

        # Find the method signature line
        signature_line_idx = None
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if line_stripped.startswith('//') or line_stripped.startswith('/*'):
                continue
            if '(' in line and ')' in line:
                signature_line_idx = i
                break

        if signature_line_idx is None:
            return method_body

        signature_line = lines[signature_line_idx]



        # Method 1: Find method name before '('
        if '(' in signature_line:
            before_paren = signature_line[:signature_line.find('(')]
            words = before_paren.strip().split()

            if words:
                # Find method name (last non-type word)
                for word in reversed(words):
                    clean_word = word.strip('*&<>[]')
                    if clean_word and clean_word not in FIMPreprocessor.JAVA_TYPES:
                        # Found potential method name
                        method_name = clean_word
                        start_idx = signature_line.rfind(method_name)
                        if start_idx != -1:
                            # Replace with <MASK>
                            masked_line = (
                                signature_line[:start_idx] +
                                "<MASK>" +
                                signature_line[start_idx + len(method_name):]
                            )
                            lines[signature_line_idx] = masked_line
                            return '\n'.join(lines)

        return method_body

    @staticmethod
    def create_fim_example(method_body, method_name):
        """
        Create FIM format training example
        Returns: (fim_input, fim_output) or (None, None) if failed
        """
        # 1. Mask the method name in the body
        masked_body = FIMPreprocessor.mask_method_signature(method_body)

        # 2. Find the <MASK> position
        mask_pos = masked_body.find("<MASK>")
        if mask_pos == -1:
            return None, None

        # 3. Split into prefix and suffix
        prefix = masked_body[:mask_pos]
        suffix = masked_body[mask_pos + 6:]  # Length of "<MASK>"

        # 4. Create FIM format input
        fim_input = (
            f"{FIMPreprocessor.FIM_PREFIX}{prefix}"
            f"{FIMPreprocessor.FIM_SUFFIX}{suffix}"
            f"{FIMPreprocessor.FIM_MIDDLE}"
        )

        # 5. Create FIM format output
        fim_output = f"{method_name}{FIMPreprocessor.END_OF_TEXT}"

        return fim_input, fim_output

    @classmethod
    def process_jsonl_file(cls, input_path, output_path, max_samples=None):
        """
        Process a JSONL file from raw format to FIM format
        """
        print(f"Processing {input_path} -> {output_path}")

        processed_count = 0
        skipped_count = 0

        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:

            # Count total lines for progress bar
            total_lines = sum(1 for _ in open(input_path, 'r', encoding='utf-8'))
            if max_samples:
                total_lines = min(total_lines, max_samples)

            for i, line in tqdm(enumerate(infile), total=total_lines, desc="Processing"):
                if max_samples and i >= max_samples:
                    break

                try:
                    data = json.loads(line.strip())
                    method_body = data.get('body', '')
                    method_name = data.get('name', '')

                    if not method_body or not method_name:
                        skipped_count += 1
                        continue

                    # Create FIM example
                    fim_input, fim_output = cls.create_fim_example(method_body, method_name)

                    if fim_input and fim_output:
                        # Save as combined text for training
                        output_data = {
                            "text": fim_input + fim_output
                        }
                        outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                        processed_count += 1
                    else:
                        skipped_count += 1

                except Exception as e:
                    skipped_count += 1
                    if i < 5:  # Print first few errors
                        print(f"  Error processing line {i}: {e}")

        print(f"✅ Processed: {processed_count}, Skipped: {skipped_count}")
        return processed_count

def main():
    """Main function for standalone execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Convert Java methods to FIM format')
    parser.add_argument('--input', required=True, help='Input JSONL file path')
    parser.add_argument('--output', required=True, help='Output JSONL file path')
    parser.add_argument('--max-samples', type=int, help='Maximum number of samples to process')

    args = parser.parse_args()

    processor = FIMPreprocessor()
    processor.process_jsonl_file(args.input, args.output, args.max_samples)

if __name__ == "__main__":
    main()
