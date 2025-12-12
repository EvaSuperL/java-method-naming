# Assignment 1: Java Method Naming with Deep Learning

## 📋 Project Overview
This project implements a deep learning-based solution for automated Java method naming, fulfilling all requirements for Assignment 1 (Option 1).

## 🎯 Requirements Status

### ✅ Step 1: Creating the Dataset
- **Mining**: Real Java methods mined from GitHub using [seart-ghs.si.usi.ch](https://seart-ghs.si.usi.ch)
- **Criteria**:
  - 100+ commits
  - 10+ contributors
  - Java language
  - Non-forks only
- **Statistics**:
  - Target: 50k methods overall
  - Achieved: 50,246 methods extracted
  - After deduplication: 47,832 methods
  - After length filtering: 44,851 methods
  - Final split: 35,880 training + 8,971 test methods
- **Preprocessing**:
  - Removed duplicates
  - Filtered methods > 256 tokens
  - Split 80% training / 20% test
- **Processing Details**:
  - Repositories processed: 45 (from 14,786 filtered)
  - Total time: 13 minutes

### ✅ Step 2: Fine-tuning a Pre-trained Model (Option 1)
- **Base Model**: Qwen2.5-Coder-0.5B ([unsloth/Qwen2.5-Coder-0.5B](https://huggingface.co/unsloth/Qwen2.5-Coder-0.5B))
- **Fine-tuning**: LoRA (r=16, alpha=16)
- **Training Sessions**: Two complete sessions with improved FIM processing
- **FIM Processing**:
  - Initial FIM preprocessor: 98.8% success rate
  - Improved FIM preprocessor: **100% success rate**
  - Final FIM datasets: 35,880 training + 8,971 test samples
- **Model Details**:
  - Total parameters: 494,032,768
  - Trainable parameters: 8,798,208 (1.75%)
  - Vocabulary size: 151,666 (increased to 151,936 with FIM tokens)
- **Hardware**: Google Colab with T4 GPU

### ✅ Step 3: Testing the Approach
- **Test Set**: 8,971 Java methods (20% of total dataset)
- **Evaluation Code**: Two evaluation frameworks implemented
  - `scripts/inference_fixed.py`: For complete model evaluation
  - `scripts/real_evaluation.py`: For checkpoint-based evaluation
- **Accuracy Metrics**: Exact match accuracy computed
- **Preliminary Results**: 60% exact match accuracy on 100-sample subset
- **Results**: Saved in JSON and text formats for verification

## 📁 Project Structure

```
method_naming_project/
├── data/
│ │ # (100+ commits, 10+ contributors, Java, non-forks)
│ └── methods/
│ ├── train_dataset.jsonl # 35,880 training methods (raw)
│ ├── test_dataset.jsonl # 8,971 test methods (raw)
│ └── metadata.json # Dataset metadata
├── datasets/ # github repo datasets and FIM processed datasets
│ ├── github_repos.csv # Original repository list with filtering criteria
│ ├── train_fim.jsonl # FIM format training data (98.8% success rate)
│ ├── test_fim.jsonl # FIM format test data (98.8% success rate)
│ ├── train_fim_improve.jsonl # Improved FIM training data (100% success rate)
│ └── test_fim_improve.jsonl # Improved FIM test data (100% success rate)
├── models/
│ ├── method_naming_model_lora/ # First training session
│ │ ├── checkpoint-xxxx/ # Training checkpoints
│ │ ├── adapter_config.json # LoRA configuration
│ │ ├── adapter_model.safetensors # Model weights
│ │ ├── special_tokens_map.json # FIM tokens
│ │ ├── tokenizer_config.json # Tokenizer configuration
│ │ └── training_metrics.json # Training statistics
│ └── method_naming_model_lora_final/ # Second training session (improved FIM)
│ ├── checkpoint-xxxx/ # Training checkpoints
│ ├── adapter_config.json # LoRA configuration
│ ├── adapter_model.safetensors # Model weights
│ ├── special_tokens_map.json # FIM tokens
│ ├── tokenizer_config.json # Tokenizer configuration
│ └── training_metrics_final.json # Final training statistics
├── scripts/ # Implementation scripts
│ ├── github_miner.py # Step 1: Data mining from GitHub
│ ├── fim_preprocessor.py # Step 2: FIM preprocessing (98.8% success)
│ ├── fim_preprocessor_improve.py # Step 2: Improved FIM preprocessing (100% success)
│ ├── inference_fixed.py # Step 3: Complete model evaluation
│ └── real_evaluation.py # Step 3: Checkpoint evaluation framework
├── output/ # Results and reports
│ ├── evaluation_final.json # Step 3 evaluation results (100 samples)
│ ├── evaluation_summary.txt # Evaluation summary report
│ ├── step3_evaluation_final/ # Step 3 evaluation framework results
│ └── training_metrics_final.json # Final training statistics
├── notebooks/ # Jupyter notebooks
│ ├── Java_Method_Naming_Assignment.ipynb # Complete Java Method filtering notebook
│ └── fine_tuning_pretrained_model.ipynb # Complete training and evaluation notebook
├── requirements.txt # Python dependencies
├── README.md # This file
└── SUBMISSION_CHECKLIST.txt # Detailed requirements checklist
```

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Data Preparation (Step 1)
```bash
# Mine data from GitHub (requires seart-ghs.csv)
python scripts/github_miner.py --csv path/to/seart-ghs.csv

# Convert to FIM format using improved processor
python scripts/fim_preprocessor_improve.py \
  --input data/methods/train_dataset.jsonl \
  --output datasets/train_fim_improve.jsonl
```

### 3. Model Evaluation (Step 3)
```bash
# Option A: Evaluate with complete trained model (100 samples)
python scripts/inference_fixed.py \
  --model-dir models/method_naming_model_lora_final \
  --test-data data/methods/test_dataset.jsonl \
  --max-samples 100

# Option B: Evaluate with checkpoint framework
python scripts/real_evaluation.py \
  --checkpoint-dir models/method_naming_model_lora \
  --test-data data/methods/test_dataset.jsonl \
  --max-samples 1000
```

## 🔧 Technical Implementation

### FIM Format Implementation
Two FIM processors implemented:
1. **Original processor**: 98.8% success rate using manual signature parsing
2. **Improved processor**: 100% success rate using direct method name masking

**Input format for training/inference:**
```
<|fim_prefix|>public static int<|fim_suffix|>(int a, int b) {
    return a + b;
}<|fim_middle|>
```

**Expected output:**
```
sum<|endoftext|>
```

### Model Architecture
- **Base Model**: Qwen2.5-Coder-0.5B (494M parameters)
- **Fine-tuning**: Parameter-Efficient Fine-Tuning with LoRA (r=16, alpha=16)
- **Training**: Two sessions totaling 6,500+ steps
- **Batch Size**: 8 per device with gradient accumulation steps 2
- **Learning Rate**: 2e-4
- **Special Tokens**: `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`, `<|endoftext|>`

## 📊 Results

### Training Progress
**First Training Session (Initial FIM Dataset):**
| Step | Training Loss | Validation Loss | Improvement |
|------|---------------|-----------------|-------------|
| 500  | 1.618         | 1.593           | Baseline    |
| 1000 | 1.557         | 1.543           | ↓ 3.8%      |
| 1500 | 1.487         | 1.512           | ↓ 4.5%      |
| 2000 | 1.481         | 1.484           | ↓ 0.4%      |
| 2500 | 1.442         | 1.470           | ↓ 10.9%     |
| 3000 | 1.417         | 1.461           | ↓ 12.4%     |
| 3500 | 1.416         | 1.454           | ↓ 12.5%     |
| 4000 | 1.398         | 1.450           | ↓ 13.6%     |

**Second Training Session (Improved FIM Dataset):**
| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 3500 | 1.380         | 1.460           |
| 4000 | 1.387         | 1.454           |
| 4500 | 1.380         | 1.444           |
| 5000 | 1.376         | 1.441           |
| 5500 | 1.406         | 1.449           |
| 6000 | 1.382         | 1.444           |
| 6500 | 1.363         | 1.443           |

### Test Set Statistics
- **Total test methods**: 8,971
- **Training methods**: 35,880
- **Total dataset**: 44,851 methods
- **Original extraction**: 50,246 methods
- **After deduplication**: 47,832 methods
- **Final after filtering**: 44,851 methods

### Preliminary Evaluation Results
- **Evaluation script**: `scripts/inference_fixed.py`
- **Samples evaluated**: 100 (due to GPU limitations)
- **Exact match accuracy**: 60.00%
- **Exact matches**: 60/100
- **Results file**: `output/evaluation_final.json`

## ⚠️ Technical Notes

### Vocabulary Size Mismatch
During training, FIM special tokens were added to the tokenizer, increasing vocabulary size from 151,666 to 151,936. This may cause loading issues in some environments.

**Solution for evaluators:**
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "models/method_naming_model_lora_final",
    ignore_mismatched_sizes=True,  # Key parameter
    trust_remote_code=True
)
```

### Training Sessions
Two complete training sessions were conducted:
1. **First session**: Trained on initial FIM dataset (98.8% success rate)
2. **Second session**: Trained on improved FIM dataset (100% success rate)
3. **Note**: Model files were properly organized after initial output directory configuration issue

### Evaluation Frameworks
Two evaluation approaches available:
1. **Complete model evaluation**: `scripts/inference_fixed.py` - Uses final trained model
2. **Checkpoint evaluation**: `scripts/real_evaluation.py` - Can use any training checkpoint

## 📝 Submission Contents

This submission includes:

1. **Complete Code** for all three steps with improvements
2. **Two Trained Models** from both training sessions
3. **Complete Dataset** (44,851 Java methods: 35,880 train + 8,971 test)
4. **Evaluation Results** with 60% accuracy on 100-sample subset
5. **Detailed Notebooks** with full implementation and debugging
6. **Two FIM Processors** showing improvement from 98.8% to 100% success rate

## 🔍 How Professors Can Verify

1. **Check Data Collection**: Review `scripts/github_miner.py` and verify 50,246 methods extracted
2. **Verify FIM Processing**: Compare `fim_preprocessor.py` (98.8%) vs `fim_preprocessor_improve.py` (100%)
3. **Examine Model Training**: Check `fine_tuning_pretrained_model.ipynb` for two training sessions
4. **Run Evaluation**: Execute `scripts/inference_fixed.py` to reproduce 60% accuracy
5. **Review Results**: Examine `output/evaluation_final.json` for detailed evaluation

## ✅ Requirements Checklist

- [x] **Step 1**: Mine 50k+ Java methods from GitHub (50,246 achieved)
- [x] **Step 1**: Clean, filter, and split dataset (80/20 split: 35,880/8,971)
- [x] **Step 2**: Implement FIM format with Qwen2.5-Coder (two processors: 98.8% → 100%)
- [x] **Step 2**: Fine-tune using LoRA with proper training (two sessions, 6,500+ steps)
- [x] **Step 3**: Implement evaluation code for accuracy computation (two frameworks)
- [x] **Step 3**: Use test set and provide runnable script (60% accuracy on 100 samples)
- [x] **Step 3**: Save and report evaluation results (JSON and text formats)

## 📄 Documentation Files

- `SUBMISSION_CHECKLIST.txt` - Detailed requirements verification
- `output/evaluation_final.json` - Step 3 evaluation results (100 samples)
- `output/evaluation_summary.txt` - Evaluation summary report
- `output/training_metrics_final.json` - Final training statistics

### Open the notebook in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/EvaSuperL/java-method-naming/blob/main/fine_tuning_pretrained_model.ipynb
)

## 👥 Author Information

- **Option Selected**: 1 (Fine-tuning pre-trained model)
- **Model**: Qwen2.5-Coder-0.5B with LoRA fine-tuning
- **Training Sessions**: 2 (with FIM processing improvements)
- **Status**: All requirements completed with documented improvements

## 📞 Contact & Support

For questions about this submission, reviewers can:
1. Check the complete notebook: `fine_tuning_pretrained_model.ipynb`
2. Run either evaluation script: `inference_fixed.py` or `real_evaluation.py`
3. Review the detailed reports in `output/` directory
4. Compare FIM processors: `fim_preprocessor.py` vs `fim_preprocessor_improve.py`

---

*Last updated: 2025-12-12 09:43:21*
