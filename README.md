# 🧠 Continuous Speech Imagery Decoding: EEG Semantic Category Classification Based on CHISCO Dataset

This project aims to decode EEG signals of Chinese continuous imagined speech, mapping brain activity to high-level semantic categories. Based on the CHISCO dataset released by Harbin Institute of Technology (currently the largest individual neural language decoding dataset), we explore the ability to decode semantic information from EEG signals during continuous imagined speech.

---

## 🎯 Research Background

### Two Paradigms of Speech Imagery Decoding

1. **Discrete Speech Imagery**
   - *Description*: Subjects imagine single vowels or simple words
   - *Characteristics*: Limited categories, relatively simple

2. **Continuous Speech Imagery**
   - *Description*: Subjects imagine complete sentences or passages
   - *Characteristics*: More natural language, more challenging

### Advantages of CHISCO Dataset
- **Largest scale**: Individual neural language decoding dataset
- **Continuous decoding**: First attempt to decode continuous imagined language
- **Semantically rich**: 6000+ daily phrases, covering 39 semantic categories
- **Chinese specific**: Focuses on neural representation of Chinese language

---

## 🔧 Data Processing Pipeline

### 1. Data Loading and Preprocessing
- Load raw CHISCO EEG data
- Apply standard preprocessing pipeline:
  - Normalize EEG signals to standardized range
  - Bad channel detection and interpolation
  - Downsampling to manageable sampling rate


### 2. Semantic Label Mapping
Raw Data → Semantic Category Mapping → High-level Category Simplification

**Mapping basis**: Use the `textmap.json` file to map each stimulus sentence to its semantic category, then merge into high-level categories.

---

## 🧬 Model Architecture

### 1. Core Model: Interpretability Gated Networks
**Interpretability Gated Networks (IGN)**

- **Core idea**: Mixture of experts + shapelet interpretability
- **Time series classification**: Specifically designed for interpretable time series analysis
- **Advantages**:
  - Automatically learns discriminative time segments (shapelets)
  - Provides interpretability for model decisions
  - Suitable for multivariate time series (e.g., EEG)

### 2. Comparison Models
| Model | Characteristics | Applicability |
|-------|----------------|---------------|
| **Transformer** | Self-attention mechanism, captures long-range dependencies | Global feature extraction for EEG sequences |
| **EEG-CNN** | Convolutional neural network, spatial-temporal feature extraction | Traditional EEG decoding baseline |

---

## 🚀 Experimental Design

### Experimental Settings
- **Task**: 3-class semantic classification (Daily Life/Emotion/Professional & Work)
- **Evaluation metrics**: Accuracy, Precision, Recall, F1-score
- **Cross-validation**: Leave-one-subject-out cross-validation
- **Comparative experiments**:
  1. IGN model vs. Transformer
  2. IGN model vs. EEG-CNN
  3. Ablation studies (validate contributions of IGN components)

### Expected Contributions
1. **Methodology**: Validate the effectiveness of IGN for continuous speech imagery decoding
2. **Interpretability**: Provide insights into neural mechanisms of EEG semantic decoding
3. **Benchmark performance**: Establish baseline results for CHISCO dataset

---



---

## 🛠️ Technical Stack
## Dataset Download

### CHISCO Dataset
- **Official Source**: OpenNeuro
- **Description**: Large-scale Chinese continuous imagined speech EEG dataset

### Example Datasets
- **Source**: Timeseries Classification Website
- **Purpose**: For testing and development

## Quick Start Guide

### Environment Setup
**Note**: The code is currently under development

### Running Experiments
For classification experiments, run:
bash ./reproduce/run_uea.sh
复制
### Steps
1. **Download dependencies**: Ensure all required packages are installed
2. **Prepare data**: Place dataset in the appropriate directory
3. **Run script**: Execute the provided shell script to start training
4. **Check results**: Monitor training progress and evaluate model performance
