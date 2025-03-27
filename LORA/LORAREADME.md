# FashionMNIST Classification with LoRA (Low-Rank Adaptation)

Improving Dress class performance using parameter-efficient fine-tuning on a pretrained Deep Neural Network.

## 📌 Overview
- **Goal**: Enhance Dress class precision/recall (class 3) without full model retraining
- **Method**: Applied LoRA (Low-Rank Adaptation) to linear layers
- **Result**: 
  - Dress precision ↑ 81% (from 71% baseline)
  - Dress recall ↑ 92% (from 94% baseline)
  - Overall accuracy: 84.8%

## 🔑 Key Features
- **LoRA Adaptation**: Added low-rank matrices to linear layers (rank=8)
- **Targeted Fine-tuning**: Focused on Dress class using:
  - 3× class oversampling
  - Class-weighted loss (weight=3.0 for Dress)
- **Parameter Efficiency**:
  - Trainable params: 30,352 (3.25% of total)
  - Non-trainable params: 903,510 (96.75%)

## 🧠 Model Architecture
```python
DeepNN(
  (linear1): LoRALayer(original_layer=Linear(784->500))
  (relu): LeakyReLU
  (linear2): LoRALayer(Original=Linear(500->1000))
  (linear3): LoRALayer(Original=Linear(1000->10))
)


