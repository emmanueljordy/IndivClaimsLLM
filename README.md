# IndivClaimsLLM

This repository contains the code for the Master's thesis in Actuarial and Financial Engineering, focusing on individual claims reserves estimation using Large Language Models (LLMs) and hybrid machine learning approaches.

## Project Structure

```
IndivClaimsLLM/
├── data/                              # Data directory
│   ├── raw/                          # Raw data files
│   │   ├── BI_10000_short.txt        # Business Interruption data
│   │   ├── EH_10000_short.txt        # Employee Handbook data  
│   │   └── data.txt                  # Main dataset
│   ├── BI/                           # Business Interruption synthetic data
│   │   ├── dat_gpt.txt
│   │   ├── dat_openelm.txt
│   │   ├── dat_qwen3.txt
│   │   └── dat_smollm2.txt
│   └── EH/                           # Employee Handbook synthetic data
├── scripts/                          # Training scripts for different LLMs
│   ├── train_gpt2_large.py
│   ├── train_openelm.py
│   ├── train_qwen3.py
│   └── train_smollm2.py
├── Rcode/                            # R and Python analysis code
│   ├── Hybrid_GBM/                   # Hybrid ML pipeline
│   │   ├── hybrid_ml_pipeline.py
│   │   ├── improved_robust_pipeline.py
│   │   ├── ml_improved_pipeline.py
│   │   └── nngabreservepython/       # Neural network reserves package
│   └── reserving_hierarchical/       # R scripts for hierarchical modeling
├── tabula_middle_padding_compromise/  # Core Tabula framework
│   ├── tabula.py                     # Main Tabula class
│   ├── tabula_trainer.py             # Custom trainer
│   ├── tabula_dataset.py             # Dataset handling
│   ├── tabula_utils.py               # Utilities
│   ├── data_preprocessing.py         # Data preprocessing
│   └── generation_processors.py     # Text generation processors
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training LLMs)
- At least 16GB RAM (32GB+ recommended for larger models)

### Python Environment Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd IndivClaimsLLM
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Additional Setup for GPU Training

For CUDA support and GPU acceleration:

```bash
# Install PyTorch with CUDA (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For Windows with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### HuggingFace Authentication (Optional)

For accessing gated models:

```bash
pip install huggingface_hub
huggingface-cli login
```

Or set the environment variable:
```bash
export HF_TOKEN="your_token_here"
```

## Usage

### 1. Training LLM Models

The repository includes scripts for training different language models on actuarial data.

#### GPT-2 Large Training
```bash
cd scripts
python train_gpt2_large.py \
    --dataset bi \
    --output_dir ../output/gpt2_large_bi \
    --logging_dir ../logs/gpt2_large_bi \
    --run_name gpt2_large_bi_experiment \
    --num_samples 10000 \
    --fresh_training
```

#### OpenELM Training
```bash
python train_openelm.py \
    --dataset eh \
    --output_dir ../output/openelm_eh \
    --logging_dir ../logs/openelm_eh \
    --run_name openelm_eh_experiment \
    --num_samples 5000 \
    --fresh_training
```

#### Available Arguments:
- `--dataset`: Choose between `bi` (Business Interruption) or `eh` (Employee Handbook)
- `--output_dir`: Directory to save trained models
- `--logging_dir`: Directory for training logs
- `--run_name`: Experiment identifier
- `--num_samples`: Number of synthetic samples to generate
- `--fresh_training`: Start training from scratch
- `--continue_training`: Continue from existing model
- `--use_qlora`: Enable QLoRA for memory-efficient training
- `--hf_token`: HuggingFace token for gated models

### 2. Hybrid ML Pipeline

The hybrid pipeline combines statistical methods with machine learning for robust reserves estimation.

```bash
cd Rcode/Hybrid_GBM
python hybrid_ml_pipeline.py
```

#### Alternative Pipelines:
```bash
# Improved robust pipeline
python improved_robust_pipeline.py

# ML-enhanced pipeline
python ml_improved_pipeline.py
```

### 3. Data Preprocessing

The Tabula framework includes sophisticated data preprocessing:

```python
from tabula_middle_padding_compromise.data_preprocessing import (
    clean_bi_dataset_for_tokenization,
    clean_eh_dataset_for_tokenization
)

# Clean BI dataset
df_clean, metadata = clean_bi_dataset_for_tokenization(your_data)
```

### 4. Custom Model Training

For advanced users, you can use the Tabula framework directly:

```python
from tabula_middle_padding_compromise.tabula import Tabula
import pandas as pd

# Load your data
data = pd.read_csv("data/raw/your_data.txt", sep=";")

# Initialize Tabula
tabula = Tabula(
    llm='gpt2-large',
    epochs=30,
    batch_size=32,
    categorical_columns=['LoB', 'cc'],
    use_lora=True
)

# Train the model
tabula.fit(data)

# Generate synthetic data
synthetic_data = tabula.sample(n_samples=1000)
```

## Configuration

### Model Parameters

Key parameters for training:

- **Learning Rate**: 1e-4 (default)
- **Batch Size**: 32 (adjust based on GPU memory)
- **Epochs**: 30 (default)
- **LoRA Rank**: 64
- **Quantization**: 4-bit (QLoRA)

### Hardware Requirements

| Model Type | Min RAM | Recommended RAM | GPU Memory |
|------------|---------|-----------------|------------|
| GPT-2 Large| 16GB    | 32GB           | 12GB       |
| OpenELM    | 8GB     | 16GB           | 8GB        |
| Qwen3      | 8GB     | 16GB           | 8GB        |
| SmolLM2    | 4GB     | 8GB            | 4GB        |

## Output

### Generated Files

Training produces several output files:

- **Models**: `output/[model_name]/model/` - Trained model files
- **Synthetic Data**: `output/[model_name]/synthetic_data_[model].csv`
- **Results**: `output/[model_name]/experiment_results.json`
- **Logs**: `logs/[model_name]/` - Training logs and metrics

### Results Analysis

The hybrid ML pipeline generates comprehensive results:

- **CSV Results**: `Results/Tables/hybrid_ml_results.csv`
- **Text Report**: `Results/Tables/hybrid_ml_report.txt`

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size: `--batch_size 16`
   - Enable gradient accumulation
   - Use QLoRA: `--use_qlora`

2. **Import Errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

3. **Data Loading Issues**:
   - Verify data file paths are correct
   - Check file permissions
   - Ensure proper file format (semicolon-separated)

4. **HuggingFace Authentication**:
   - Set HF_TOKEN environment variable
   - Use `--hf_token` argument
   - Run `huggingface-cli login`

### Performance Optimization

- Use mixed precision training for faster training
- Enable flash attention for memory efficiency
- Consider model parallelism for very large models
- Use SSD storage for faster data loading

## Research Context

This code implements novel approaches for actuarial reserve estimation:

1. **LLM-based Synthetic Data Generation**: Using transformer models to generate realistic claim data
2. **Hybrid Statistical-ML Methods**: Combining traditional actuarial methods with modern ML
3. **Domain-Specific Preprocessing**: Tailored data processing for actuarial applications
4. **Robust Evaluation Framework**: Comprehensive testing across different model architectures

## Contributing

When contributing to this project:

1. Follow the existing code structure
2. Add appropriate documentation
3. Test with multiple datasets
4. Update requirements.txt for new dependencies

## License

This project is part of academic research in Actuarial and Financial Engineering.

## Contact

For questions about this research, please refer to the associated thesis documentation.