# 💡 QLoRA + Pairwise: Optimized Decentralized Training System

This project enhances the [Hivemind](https://github.com/learning-at-home/hivemind)-based decentralized training system **DeDLOC** by integrating two key techniques:
- **QLoRA**: Quantized Low-Rank Adaptation for efficient fine-tuning
- **Pairwise averaging**: A robust communication strategy for decentralized peer networks

> ✅ This is the default branch containing the **final merged implementation** of QLoRA and Pairwise.

---

## 🔧 Key Features

### 1. QLoRA Integration (Memory-Efficient Fine-Tuning)
- Applies 8-bit quantization using `bitsandbytes`  
- Uses LoRA adapters via the `peft` library to train only a subset of parameters  
- Reduces GPU memory usage by over **20%**
- Achieves fast convergence with minimal accuracy loss

### 2. Pairwise Communication Strategy (Network Robustness)
- Replaces global all-reduce with **2-peer group-based local averaging**
- Employs a **Leader–Follower** structure within each group to avoid deadlocks
- Demonstrated **stable training even when some peers were disconnected**

### 3. Integrated Distributed Training Pipeline
- Docker-based setup with 1 coordinator and multiple workers  
- Integrated with wandb for live logging of accuracy, loss, memory usage, and throughput  
- Fine-tuning performed on BERT-Tiny using the WikiText-103 dataset

## 🔀 Other Related Branches

- [`hivemind`](https://github.com/WKJ-00/hivemind):  
  ➤ Original DeDLOC + Hivemind-only implementation

- [`stellatrain`](https://github.com/your_repo/tree/stellatrain):  
  ➤ Experimental branch for Partial Staleness & compression techniques

---

## 📊 Summary of Results

| Configuration         | Accuracy | Final Loss | GPU Memory | Convergence Time | Key Characteristics                          |
|-----------------------|----------|------------|-------------|------------------|-----------------------------------------------|
| Baseline (DeDLOC)     | 0.4155   | 3.91       | ~3.6GB      | 110 hours        | Full training, stable convergence             |
| QLoRA only            | 0.3973   | 4.13       | ~2.85GB     | 10 hours         | Lightweight fine-tuning, fastest convergence  |
| QLoRA + Pairwise      | 0.4266   | 4.16       | ~2.9GB      | 70 hours         | Best balance of accuracy, memory & robustness |

---

## 🚀 Quick Start

```bash
# 1. [Optional] Create virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare dataset and tokenizer
python tokenize_wikitext103.py
python generate_eval_subset.py

# 4. Launch coordinator (first peer)
python run_first_peer.py

# 5. Launch worker peers
python run_trainer.py
