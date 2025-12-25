# FreqAgent

> A spectral analysis framework for understanding agent trajectories through frequency-domain analysis

## Overview

FreqAgent is a sophisticated research framework designed to analyze the spectral characteristics of agent trajectories in large language models. By applying frequency-domain transforms (DCT, FFT) to semantic segments of agent behavior, this framework reveals hidden patterns in how models transition between thinking and tool-assisted reasoning.

## Key Features

- **Multi-Scale Spectral Analysis**: Analyze agent behavior at token, segment, and block levels
- **Semantic Segmentation**: Automatic parsing of trajectories into Think, Tool Call, and Tool Response phases
- **Frequency-Domain Metrics**: Compute spectral entropy, high-frequency ratio, and energy concentration
- **Entropy-Based Analysis**: Quantitative entropy analysis across transformer layers
- **Attention Visualization**: Block-level attention pattern analysis and visualization
- **Real Model Integration**: Support for WebSailor-3B and other model architectures

## Project Structure

```
FreqAgent/
├── main.py                        # Core spectral analysis framework
├── DCT_II/                        # DCT-II frequency analysis implementation
│   └── kv_frequency_analysis.py  # KV cache frequency analysis
├── Entropy_Attention_Map/         # Entropy and attention analysis
│   ├── Entropy/                  # Layer-wise entropy statistics
│   └── Attention_Map/            # Block-level attention visualization
├── Entropy_sglang_logits/        # SGLang-based entropy analysis
├── websailor3B/                  # WebSailor-3B model integration
├── websailor3B_json/             # JSON trajectory data
└── output_DCT_II/                # Analysis outputs and visualizations
```

## Installation

### Requirements

- Python 3.8+
- PyTorch
- Transformers
- Scientific computing libraries

### Setup

```bash
# Clone the repository
git clone https://github.com/ZhenfengSu/FreqAgent.git
cd FreqAgent

# Install dependencies
pip install numpy matplotlib scipy torch transformers seaborn tqdm
```

## Usage

### Basic Spectral Analysis

```bash
python main.py
```

This will:
1. Parse agent trajectories from JSON files
2. Segment trajectories into semantic phases (Think, Tool Call, Tool Response)
3. Apply DCT/FFT transforms
4. Compute spectral metrics
5. Generate comprehensive visualizations

### Advanced Frequency Analysis

```bash
# KV cache frequency analysis
python DCT_II/kv_frequency_analysis.py

# Entropy-based analysis
python Entropy_Attention_Map/Entropy/entropy_analysis.py

# Attention map visualization
python Entropy_Attention_Map/Attention_Map/block_attention.py
```

## Core Components

### 1. Spectral Analysis Framework (`main.py`)

Analyzes agent trajectories using frequency-domain transforms:

- **Spectral Entropy**: Measures randomness and complexity of behavior patterns
- **High-Frequency Ratio**: Quantifies rapid state transitions
- **Energy Concentration**: Identifies focused vs. dispersed cognitive patterns

### 2. DCT-II Frequency Analysis (`DCT_II/`)

Advanced frequency-domain analysis of KV caches:

- Semantic slicing of KV cache data
- Fixed-length resampling for frequency bin alignment
- Power spectrum density calculations
- Comparative analysis of thinking vs. tool-calling patterns

### 3. Entropy Analysis (`Entropy_Attention_Map/Entropy/`)

Layer-wise entropy statistics across transformer layers:

- Phase-wise statistics (Thinking, Action, Post-Response phases)
- Attention proportion scoring
- Ablation study comparisons
- "Action Activation" phenomenon quantification

### 4. Attention Map Visualization (`Entropy_Attention_Map/Attention_Map/`)

Block-level attention pattern analysis:

- Parses complex agent dialogues into semantic blocks
- Aggregates attention weights across multiple heads
- Generates publication-quality heatmaps
- Cross-layer attention pattern comparison

## Methodology

The analysis pipeline follows these steps:

1. **Data Acquisition**: Extract agent trajectories with semantic segmentation
2. **Preprocessing**: Token counting, embedding generation, normalization
3. **Frequency Transformation**: Apply DCT/FFT to convert time-domain to frequency-domain
4. **Metric Calculation**: Compute spectral characteristics for each segment type
5. **Visualization**: Generate comprehensive charts and plots
6. **Analysis**: Compare patterns across segment types and model layers

## Research Applications

- **Cognitive Pattern Analysis**: Understand how models transition between reasoning and action
- **Model Interpretability**: Reveal hidden patterns in agent behavior
- **Ablation Studies**: Compare model behavior under different conditions
- **Architecture Analysis**: Examine layer-specific characteristics
- **Tool Usage Research**: Study the thinking-action boundary in tool-assisted LLMs

## Key Findings

The framework enables investigation of:
- Spectral differences between thinking and tool-calling patterns
- Characteristic frequency signatures of different agent behaviors
- Layer-wise activation patterns during semantic transitions
- Attention flow between semantic blocks
- Entropy changes across different phases of agent trajectories

## Dependencies

```
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
torch>=1.9.0
transformers>=4.0.0
seaborn>=0.11.0
tqdm>=4.62.0
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{freqagent2024,
  title={FreqAgent: A Spectral Analysis Framework for Agent Trajectories},
  author={Su, Zhenfeng},
  year={2024},
  url={https://github.com/ZhenfengSu/FreqAgent}
}
```

## License

This project is licensed under the MIT License.

## Contact

For questions and feedback, please open an issue on GitHub or contact [Zhenfeng Su](https://github.com/ZhenfengSu).

---

**Note**: This framework is designed for research in mechanistic interpretability and understanding the internal dynamics of large language model agents.
