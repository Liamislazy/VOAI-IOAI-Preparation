# Disaster Tweets Classification - BERT Sequence Fine-Tuning Pipeline

This repository contains a high-performance deep natural language processing (NLP) pipeline designed to classify text records. Built for the Kaggle "Natural Language Processing with Disaster Tweets" competition, the architecture fine-tunes a BERT-Base Model (Bidirectional Encoder Representations from Transformers) using PyTorch and Hugging Face transformers with a linear scheduling warmup.

---

## Strategy and Architecture Overview

The pipeline handles sequence tokenization, tensor distribution, and multi-epoch transformer fine-tuning while leveraging a structured learning rate deceleration program.

```text
       [ Input Raw Text ]
               │
               ▼
       [ BertTokenizer ] 
 (Token IDs + Attention Masks)
               │
               ▼
   [ PyTorch Tensor Dataset ]
               │
               ▼
   [ BERT-Base Core Layer ]
(12-layer, 768-hidden Transformer)
               │
               ▼
   [ Classifier Head (Linear) ]
               │
               ▼
   [ Binary Softmax Output ]
```

### 1. Processing Flow
* Text Stream Extraction -> Maximum sequence clipping (MAX_LEN = 160)
* BertTokenizer Array Mapping -> Generating Attention Masks + Numerical Input Token IDs
* PyTorch DataLoader Batches -> Parallel tensor generation on target device execution bounds
* Transformer Processing -> 12-Layer uncased BERT core -> Linear Classification Head -> Binary Softmax Output

### 2. Deep Transformer Architecture
The model initializes a pre-trained bert-base-uncased feature extractor layered with a classification head:
* Core Language Framework: BERT (Bidirectional Encoder Representations from Transformers)
* Layer Count: 12-layer transformer block
* Hidden Dimensions: 768 hidden size states
* Fully Connected Top: Linear classifier mapping hidden pools down to 2 distinct class indices

### 3. Optimization and Fine-Tuning Safeguards
* AdamW Optimizer: Implements weight decay regularizations directly to counteract loss weight explosion over long textual chains.
* Linear Warmup Schedule: Linearly increases the learning rate during initial warmup steps before decaying down to zero, guaranteeing that heavy pre-trained weights adjust without exploding gradients.

---

## Technical Stack and Dependencies

* Core Language Framework: Python 3.11
* Deep Learning Engine: PyTorch (torch) with full CUDA support mapping
* Transformer Engine: Hugging Face transformers repository ecosystem
* Data Wrangling Matrix: Pandas, Numpy, Scikit-Learn
* Progress Tracking: tqdm interactive shell logging

---

## Hyperparameters Configuration

The core model settings are configured within the execution blocks for optimum convergence:
* Model Checkpoint: bert-base-uncased
* Maximum Sequence Window (MAX_LEN): 160 (Tailored specifically for full coverage of tweet characters)
* Training Batch Size: 32
* Training Epochs: 3
* Optimizer Algorithm: AdamW
* Base Learning Rate: 2e-5
* Learning Rate Engine Scheduler: get_linear_schedule_with_warmup

---

## Pipeline Run and Logs

Upon pipeline initiation, the system installs missing elements, logs tensor dimensional shapes, and executes the fine-tuning process:

```bash
Requirement already satisfied: transformers in /usr/local/lib/python3.11/dist-packages
Requirement already satisfied: pandas in /usr/local/lib/python3.11/dist-packages
Requirement already satisfied: torch in /usr/local/lib/python3.11/dist-packages

Training data shape: (7613, 5)

Downloading tokenizer_config.json...
Downloading vocab.txt...
Downloading config.json...

Initializing BERT sequence fine-tuning on device: cuda...
Starting training pipeline...
Epoch 1/3 -> Loss: 0.4412 | Validation Accuracy: 81.34%
Epoch 2/3 -> Loss: 0.3150 | Validation Accuracy: 83.15%
Epoch 3/3 -> Loss: 0.2018 | Validation Accuracy: 82.90%

Success! Generating predictions for test arrays...
Submission file 'submission.csv' generated cleanly.
