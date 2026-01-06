# project_C5045703
# Visual Storytelling with Multimodal Sequence Models (TensorFlow)

This project implements a visual storytelling system that generates a coherent
natural-language narrative conditioned on a sequence of images.
The model is trained using supervised learning with teacher forcing and is evaluated
using both diversity-based and reference-based metrics.

The goal of the project is to analyze the limitations of likelihood-based training
for storytelling and establish a strong baseline for later reinforcement learning
extensions.

---

## 1. Task Description

Visual storytelling involves generating a multi-sentence story from a sequence of
images while maintaining semantic grounding, temporal coherence, and linguistic fluency.
Unlike single-image captioning, storytelling requires modeling relationships across
multiple visual inputs and producing a coherent narrative.

This project focuses on:
- Learning a multimodal representation of image sequences and text
- Generating a final story sentence conditioned on visual context
- Evaluating narrative diversity and correctness

---

## 2. Dataset

We use the HuggingFace dataset:

**`daniel3303/StoryReasoning`**

Each example consists of:
- A sequence of images (`seq_len = 3`)
- A sequence of captions or story sentences
- The last caption is treated as the target story to generate

Preprocessing includes:
- Image resizing and normalization to `[-1, +1]`
- Tokenization using `bert-base-uncased`
- Padding and truncation to fixed sequence lengths

---

## 3. Model Architecture

The model is a multimodal sequence-to-sequence architecture:

### Visual Encoder
- Lightweight CNN
- Produces a fixed-dimensional visual feature vector per image

### Text Encoder
- LSTM encoder
- Encodes each caption into a fixed-size representation

### Temporal Modeling
- LSTM over fused visual and textual features across time
- Produces a context vector from the last timestep

### Text Decoder
- LSTM decoder conditioned on the temporal context
- Generates the target story sentence token-by-token

### Auxiliary Image Loss
- Predicts visual feature embedding of the last image
- Encourages better visual grounding during training

---

## 4. Training

Training is performed using **teacher forcing** with a combined loss:


Key settings:
- Optimizer: Adam
- Gradient clipping
- Optional freezing of the visual encoder for speed
- CPU or GPU compatible

---

## 5. Evaluation Metrics

The following metrics are computed automatically:

### Linguistic Diversity
- **Distinct-1**: ratio of unique unigrams
- **Distinct-2**: ratio of unique bigrams

### Reference-Based Metrics
- **BLEU (sentence-level, smoothed)**
- **METEOR**

Evaluation is performed using greedy decoding on the validation set.

---

## 6. Visualizations

The project includes qualitative visualizations showing:
- Input image sequences
- Ground-truth story
- Model-predicted story

These visualizations demonstrate semantic grounding and narrative coherence.

---

## 7. Repository Structure

project_root/
├── README.md
├── requirements.txt
├── config.yaml
├── experiments.ipynb
├── src/
│ ├── train.py
│ ├── model.py
│ └── utils.py
└── results/
├── checkpoints/
├── final_metrics.json

---

## 8. Running the Code

### Install dependencies
```bash
pip install -r requirements.txt
### Train the model
python src/train.py --config config.yaml
### Run experiments and visualizations
jupyter notebook experiments.ipynb
