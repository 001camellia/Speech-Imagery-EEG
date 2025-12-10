## 🧠 Continuous Speech Imagery Decoding: EEG Semantic Category Classification Based on CHISCO Dataset
This project aims to decode EEG signals of Chinese continuous imagined speech, mapping brain activity to high-level semantic categories. Based on the CHISCO​ dataset released by Harbin Institute of Technology (currently the largest individual neural language decoding dataset), we explore the ability to decode semantic information from EEG signals during continuous imagined speech.


## 🎯 Research Background

### Two Paradigms of Speech Imagery Decoding

| Paradigm | Description | Characteristics |
|----------|-------------|-----------------|
| **Discrete Speech Imagery** | Subjects imagine single vowels or simple words | Limited categories, relatively simple |
| **Continuous Speech Imagery** | Subjects imagine complete sentences or passages | More natural language, more challenging |

###Advantages of CHISCO Dataset
Largest scale: Individual neural language decoding dataset
Continuous decoding: First attempt to decode continuous imagined language
Semantically rich: 6000+ daily phrases, covering 39 semantic categories
Chinese specific: Focuses on neural representation of Chinese language
##🔧 Data Processing Pipeline
1. Data Loading and Preprocessing
Load raw CHISCO EEG data
Apply standard preprocessing pipeline:
Band-pass filtering (0.5-45 Hz)
Bad channel detection and interpolation
Artifact removal (ICA)
Re-referencing (average reference)
2. Semantic Label Mapping
Raw Data → Semantic Category Mapping → High-level Category Simplification
# Label mapping illustration
39 original semantic categories → 3 high-level categories:
1. Daily Life
2. Emotion
3. Professional & Work
Mapping basis: Use the textmap.jsonfile to map each stimulus sentence to its semantic category, then merge into high-level categories.
##🧬 Model Architecture
1. Core Model: Interpretability Gated Networks
Interpretability Gated Networks (IGN)
Core idea: Mixture of experts + shapelet interpretability
Time series classification: Specifically designed for interpretable time series analysis
Advantages:
Automatically learns discriminative time segments (shapelets)
Provides interpretability for model decisions
Suitable for multivariate time series (e.g., EEG)
2. Comparison Models
Model
Characteristics
Applicability
Transformer​
Self-attention mechanism, captures long-range dependencies
Global feature extraction for EEG sequences
EEG-CNN​
Convolutional neural network, spatial-temporal feature extraction
Traditional EEG decoding baseline
##🚀 Experimental Design
Experimental Settings
Task: 3-class semantic classification (Daily Life/Emotion/Professional & Work)
Evaluation metrics: Accuracy, Precision, Recall, F1-score
Cross-validation: Leave-one-subject-out cross-validation
Comparative experiments:
IGN model vs. Transformer
IGN model vs. EEG-CNN
Ablation studies (validate contributions of IGN components)
Expected Contributions
Methodology: Validate the effectiveness of IGN for continuous speech imagery decoding
Interpretability: Provide insights into neural mechanisms of EEG semantic decoding
Benchmark performance: Establish baseline results for CHISCO dataset
