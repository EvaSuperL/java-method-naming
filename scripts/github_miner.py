# scripts/github_miner.py
"""
GitHub Java Method Miner - Assignment 1 Step 1
Extract Java methods from repositories listed in SEART-GHS CSV to create <method_body, method_name> pairs
"""

import pandas as pd
import subprocess
import os
import json
import re
import shutil
from tqdm import tqdm
import hashlib
from datetime import datetime
from sklearn.model_selection import train_test_split

# Install required packages for Tree-sitter
import subprocess
import sys

def install_package(package_name):
    """Install a Python package if not available"""
    try:
        __import__(package_name.split('-')[0].replace('_', ''))
        return True
    except ImportError:
        print(f"Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return True
        except:
            print(f"Failed to install {package_name}")
            return False

# Install tree-sitter and java parser
install_package("tree-sitter")
install_package("tree-sitter-java")

# Now import after installation
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava

class GitHubJavaMiner:
    """
    GitHub Java method miner
    Extracts Java methods from repositories listed in SEART-GHS CSV
    """

    def __init__(self, csv_path):
        """
        Initialize the miner

        Args:
            csv_path: Path to SEART-GHS CSV file
        """
        # Verify CSV path
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        self.csv_path = csv_path

        # Use project root from environment
        project_root = os.environ.get('PROJECT_ROOT', os.getcwd())

        # Setup directories
        self.data_dir = os.path.join(project_root, "data")
        self.repos_dir = os.path.join(self.data_dir, "repositories")
        self.methods_dir = os.path.join(self.data_dir, "methods")

        for dir_path in [self.data_dir, self.repos_dir, self.methods_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Target number of methods
        self.target_methods = 50000

        # Setup Tree-sitter parser
        self.setup_parser()

        # Statistics - initialize properly
        self.stats = {
            'total_repos_in_csv': 0,
            'repos_after_filtering': 0,
            'repos_processed': 0,
            'total_methods_extracted': 0,
            'methods_after_deduplication': 0,
            'methods_after_length_filter': 0,
            'train_methods': 0,
            'test_methods': 0,
            'target_reached': False
        }

    def setup_parser(self):
        """Setup Tree-sitter Java parser"""
        try:
            self.JAVA_LANGUAGE = Language(tsjava.language())
            self.parser = Parser(self.JAVA_LANGUAGE)
            print("✅ Tree-sitter Java parser initialized")
        except Exception as e:
            print(f"❌ Tree-sitter initialization failed: {e}")
            self.parser = None

    def load_repository_list(self):
        """
        Load and filter repository list from CSV

        Returns:
            List of repository URLs
        """
        print(f"
📊 Loading repository list: {self.csv_path}")

        try:
            df = pd.read_csv(self.csv_path)
            self.stats['total_repos_in_csv'] = len(df)
            print(f"  Original repositories in CSV: {self.stats['total_repos_in_csv']}")

            # Apply filters as specified in assignment
            if "mainLanguage" in df.columns:
                df = df[df["mainLanguage"].astype(str).str.lower() == "java"]
                print(f"  After Java language filter: {len(df)}")

            if "commits" in df.columns:
                df = df[df["commits"] >= 100]
                print(f"  After commits >= 100 filter: {len(df)}")

            if "contributors" in df.columns:
                df = df[df["contributors"] >= 10]
                print(f"  After contributors >= 10 filter: {len(df)}")

            if "isFork" in df.columns:
                df = df[df["isFork"] == False]
                print(f"  After non-fork filter: {len(df)}")

            # Construct GitHub URLs
            if "name" not in df.columns:
                raise ValueError("CSV must contain 'name' column")

            repo_urls = [
                f"https://github.com/{repo_name.strip()}.git"
                for repo_name in df["name"].astype(str)
            ]

            self.stats['repos_after_filtering'] = len(repo_urls)
            print(f"  Total repositories after filtering: {self.stats['repos_after_filtering']}")

            # Process more repositories to reach 50k methods
            # Based on previous results, 50 repos gave 35k methods, so changed to ~70 repos
            needed_repos = min(70, len(repo_urls))
            print(f"  Will process {needed_repos} repositories to reach target")
            return repo_urls[:needed_repos]

        except Exception as e:
            print(f"❌ Failed to load CSV: {e}")
            return []

    def clone_repository(self, url):
        """Clone a single repository"""
        repo_name = url.split("/")[-1].replace(".git", "")
        dst = os.path.join(self.repos_dir, repo_name)

        # Skip if already exists
        if os.path.exists(dst):
            print(f"  ⏭️  Already exists: {repo_name}")
            return dst

        print(f"  📥 Cloning: {repo_name}")

        try:
            result = subprocess.run(
                [
                    "git", "clone",
                    "--depth", "1",
                    "--single-branch",
                    url, dst
                ],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                print(f"    ✅ Cloned successfully")
                return dst
            else:
                print(f"    ❌ Clone failed: {result.stderr[:200]}")
                return None

        except subprocess.TimeoutExpired:
            print(f"    ⏱️  Timeout")
            return None

    def extract_methods_from_file(self, file_path):
        """Extract methods from a Java file using Tree-sitter"""
        methods = []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()

            if not code.strip():
                return methods

            tree = self.parser.parse(bytes(code, "utf8"))
            root = tree.root_node

            # Method node types in Tree-sitter Java
            method_types = ["method_declaration", "constructor_declaration"]

            def get_text(node):
                return code[node.start_byte:node.end_byte]

            def traverse(node):
                if node.type in method_types:
                    # Extract method name
                    name_node = node.child_by_field_name("name")
                    if not name_node:
                        return

                    method_name = get_text(name_node).strip()

                    # Skip very short method names (likely incomplete)
                    if len(method_name) < 2:
                        return

                    # Extract complete method
                    full_method = get_text(node).strip()

                    methods.append({
                        "name": method_name,
                        "body": full_method,
                        "file": file_path
                    })

                for child in node.children:
                    traverse(child)

            traverse(root)

        except Exception as e:
            print(f"    ⚠️  Parsing failed {file_path}: {e}")

        return methods

    def process_repository(self, repo_path):
        """Process a single repository to extract methods"""
        methods = []

        # Find Java files
        java_files = []
        for root, dirs, files in os.walk(repo_path):
            # Skip test directories
            skip_dirs = ['test', 'tests', 'Test', 'Tests']
            dirs[:] = [d for d in dirs if d not in skip_dirs]

            for file in files:
                if file.endswith('.java'):
                    # Skip test files
                    if 'test' in file.lower() or 'Test' in file:
                        continue
                    java_files.append(os.path.join(root, file))

        if not java_files:
            print(f"    ℹ️  No Java files found")
            return methods

        print(f"    📄 Found {len(java_files)} Java files")

        # Process files - increase from 100 to 200 to get more methods
        max_files = min(200, len(java_files))
        processed_files = 0

        for i, java_file in enumerate(java_files[:max_files]):
            file_methods = self.extract_methods_from_file(java_file)
            methods.extend(file_methods)
            processed_files += 1

            # Progress update
            if processed_files % 50 == 0:
                print(f"    📝 Processed {processed_files}/{max_files} files, extracted {len(methods)} methods so far")

        print(f"    ✅ Extracted {len(methods)} methods from {processed_files} files")
        return methods

    def deduplicate_methods(self, methods):
        """Remove duplicate methods based on method body hash"""
        seen = set()
        unique_methods = []

        for method in methods:
            method_hash = hashlib.md5(method['body'].encode()).hexdigest()
            if method_hash not in seen:
                seen.add(method_hash)
                unique_methods.append(method)

        return unique_methods

    def filter_long_methods(self, methods):
        """Filter methods longer than 256 tokens"""
        try:
            # Try to import transformers for tokenization
            from transformers import AutoTokenizer

            # Use Qwen tokenizer as specified in assignment
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")

            filtered = []
            print(f"  Filtering methods with > 256 tokens...")

            for i, method in enumerate(tqdm(methods, desc="Token filtering")):
                tokens = tokenizer.encode(method['body'], add_special_tokens=False)
                if len(tokens) <= 256:
                    filtered.append(method)

                # Progress update
                if i % 10000 == 0 and i > 0:
                    print(f"    Processed {i}/{len(methods)} methods, {len(filtered)} passed filter")

            return filtered

        except Exception as e:
            print(f"⚠️  Could not use tokenizer for filtering: {e}")
            print("Using simple character count filter instead")

            # Simple fallback: filter methods with more than 2000 characters
            filtered = []
            for method in methods:
                if len(method['body']) <= 2000:
                    filtered.append(method)

            return filtered

    def save_dataset(self, data, filename):
        """Save dataset as JSONL format"""
        output_path = os.path.join(self.methods_dir, filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"💾 Saved {len(data)} items to {filename}")
        return output_path

    def run(self):
        """
        Run the complete data mining pipeline

        Returns:
            (train_methods, test_methods) or None if failed
        """
        print("=" * 60)
        print("🚀 Starting GitHub Java method mining")
        print("=" * 60)

        start_time = datetime.now()

        try:
            # 1. Load repository list
            repo_urls = self.load_repository_list()

            if not repo_urls:
                print("❌ No valid repository URLs found")
                return None

            # 2. Process repositories
            all_methods = []
            print(f"
🔍 Processing {len(repo_urls)} repositories...")

            for i, repo_url in enumerate(repo_urls):
                # Check if we've reached target
                if len(all_methods) >= self.target_methods:
                    print(f"🎯 Reached target of {self.target_methods} methods")
                    self.stats['target_reached'] = True
                    break

                # Clone repository
                repo_path = self.clone_repository(repo_url)
                if not repo_path:
                    continue

                # Extract methods
                try:
                    methods = self.process_repository(repo_path)
                    all_methods.extend(methods)
                    self.stats['repos_processed'] += 1

                    current_total = len(all_methods)
                    remaining = self.target_methods - current_total
                    print(f"  📊 Repo {i+1}/{len(repo_urls)}: +{len(methods)} methods, Total: {current_total}, Remaining: {max(0, remaining)}")

                except Exception as e:
                    print(f"  ❌ Processing failed: {e}")

                # Clean up
                try:
                    shutil.rmtree(repo_path, ignore_errors=True)
                except:
                    pass

            # Record total extracted methods
            self.stats['total_methods_extracted'] = len(all_methods)

            # 3. Data cleaning
            print(f"
🧹 Data cleaning...")
            print(f"  Total methods extracted: {self.stats['total_methods_extracted']}")

            if not all_methods:
                print("❌ No methods extracted")
                return None

            # Deduplication
            unique_methods = self.deduplicate_methods(all_methods)
            self.stats['methods_after_deduplication'] = len(unique_methods)
            print(f"  After deduplication: {self.stats['methods_after_deduplication']}")

            # Filter long methods (max 256 tokens)
            filtered_methods = self.filter_long_methods(unique_methods)
            self.stats['methods_after_length_filter'] = len(filtered_methods)
            print(f"  After length filtering: {self.stats['methods_after_length_filter']}")

            if not filtered_methods:
                print("❌ No methods after filtering")
                return None

            # Check if we have enough methods
            if self.stats['methods_after_length_filter'] < 10000:
                print(f"⚠️  Warning: Only {self.stats['methods_after_length_filter']} methods after filtering")
                print("   Consider processing more repositories or adjusting filters")

            # 4. Split dataset (80% train, 20% test)
            print(f"
📊 Splitting dataset (80% train, 20% test)...")
            train_methods, test_methods = train_test_split(
                filtered_methods,
                test_size=0.2,
                random_state=42,
                shuffle=True
            )

            self.stats['train_methods'] = len(train_methods)
            self.stats['test_methods'] = len(test_methods)

            print(f"  Training set: {self.stats['train_methods']} methods")
            print(f"  Test set: {self.stats['test_methods']} methods")

            # 5. Save datasets
            print(f"
💾 Saving datasets...")
            train_path = self.save_dataset(train_methods, "train_dataset.jsonl")
            test_path = self.save_dataset(test_methods, "test_dataset.jsonl")

            # 6. Save metadata
            metadata = {
                "statistics": self.stats,
                "target_methods": self.target_methods,
                "train_size": len(train_methods),
                "test_size": len(test_methods),
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration": str(datetime.now() - start_time),
                "note": "Assignment 1 Step 1: Data collection for Java method naming"
            }

            metadata_path = os.path.join(self.methods_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # 7. Print summary
            print("=" * 60)
            print("✅ Data mining completed!")
            print("=" * 60)
            print(f"⏱️  Total time: {datetime.now() - start_time}")
            print(f"📊 Final statistics:")
            print(f"  - Repositories in CSV: {self.stats['total_repos_in_csv']}")
            print(f"  - Repositories after filtering: {self.stats['repos_after_filtering']}")
            print(f"  - Repositories processed: {self.stats['repos_processed']}")
            print(f"  - Total methods extracted: {self.stats['total_methods_extracted']}")
            print(f"  - After deduplication: {self.stats['methods_after_deduplication']}")
            print(f"  - After length filtering: {self.stats['methods_after_length_filter']}")
            print(f"  - Training set: {self.stats['train_methods']}")
            print(f"  - Test set: {self.stats['test_methods']}")
            print(f"  - Target reached: {self.stats['target_reached']}")
            print(f"📁 Output location: {self.methods_dir}")
            print("=" * 60)

            return train_methods, test_methods

        except Exception as e:
            print(f"❌ Error during mining: {e}")
            import traceback
            traceback.print_exc()
            return None

# Main execution when run as script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='GitHub Java Method Miner')
    parser.add_argument('--csv', required=True, help='Path to SEART-GHS CSV file')

    args = parser.parse_args()

    # Create miner and run
    miner = GitHubJavaMiner(csv_path=args.csv)
    result = miner.run()

    if result:
        print("✅ Mining completed successfully!")
    else:
        print("❌ Mining failed")
        sys.exit(1)
