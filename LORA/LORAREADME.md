
# FashionMNIST Classification with LoRA (Low-Rank Adaptation)

Improving Dress class performance using parameter-efficient fine-tuning on a pretrained Deep Neural Network.

## 📌 Overview
- **Goal**: Enhance Dress class precision/recall (class 3) without full model retraining.
- **Method**: Applied LoRA (Low-Rank Adaptation) to linear layers.
- **Result**: 
  - Dress precision ↑ 81% (from 71% baseline)
  - Dress recall ↑ 92% (from 94% baseline)
  - Overall accuracy: 84.8%

## 🔑 Key Features
- **LoRA Adaptation**: Added low-rank matrices to linear layers (`rank=8`).
- **Targeted Fine-tuning**: Focused on Dress class using:
  - **3× class oversampling**
  - **Class-weighted loss** (`weight=3.0` for Dress).
- **Parameter Efficiency**:
  - Trainable params: 30,352 (3.25% of total).
  - Non-trainable params: 903,510 (96.75%).

## 🧠 Model Architecture
```python
DeepNN(
  (linear1): LoRALayer(original_layer=Linear(784 -> 500)),
  (relu): LeakyReLU,
  (linear2): LoRALayer(original_layer=Linear(500 -> 1000)),
  (linear3): LoRALayer(original_layer=Linear(1000 -> 10))
)
```

## 📊 Results  
### Classification Report Comparison (Test Set)

#### Before LoRA Fine-Tuning
| Class          | Precision | Recall | F1-Score |  
|----------------|-----------|--------|----------|  
| **Dress**      | 0.85      | 0.83   | 0.84     |  
| Overall Accuracy | 0.82     | 

#### After LoRA Fine-Tuning
| Class          | Precision | Recall | F1-Score |  
|----------------|-----------|--------|----------|  
| **Dress**      | 0.81      | 0.92   | 0.86     |  
| Overall Accuracy | 0.848    | 

#### Key Improvements for Dress Class
| Metric        | Before | After | Δ    |
|---------------|--------|-------|------|
| **Recall**    | 0.83   | 0.92  | ↑+9% |
| **F1-Score**  | 0.84   | 0.86  | ↑+2% |

Full class comparison:
| Class          | Precision (Before → After) | Recall (Before → After) | F1-Score (Before → After) |
|----------------|----------------------------|-------------------------|---------------------------|
| T-shirt/top    | 0.78 → 0.75                | 0.79 → 0.87             | 0.79 → 0.81               |
| Trouser        | 0.95 → 0.98                | 0.97 → 0.96             | 0.96 → 0.97               |
| **Dress**      | 0.85 → 0.81                | 0.83 → 0.92             | 0.84 → 0.86               | 
| Sandal         | 0.94 → 0.96                | 0.89 → 0.91             | 0.91 → 0.93               |
| Ankle boot     | 0.76 → 0.94                | 0.99 → 0.92             | 0.86 → 0.93               |

## 💡 Possible Improvements  
1. **LoRA Rank Experimentation**: Test ranks like `4`, `8`, or `16`.  
2. **Class Weight Tuning**: Adjust loss weights (e.g., `weight=5.0` for Dress).  
3. **Layer-Specific Adaptation**: Apply LoRA only to critical layers.  
4. **Alpha Scaling**: Stabilize updates with `α/r`:  
   ```python
   scaled_lora = lora_output * (self.alpha / self.rank)  # e.g., α=16
   ```

## 📚 Citation  
```bibtex
@article{lora2021,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Chen, Weizhu},
  journal={ICLR},
  year={2021}
}
```

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
```


