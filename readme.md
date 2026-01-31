# ROAR: Robust Counterfactual Explanations
## Acknowledgments & References

This project builds upon and extends the following research and implementations:

* **ROAR (Robust and Reliable Algorithmic Recourse):** The core robust optimization logic and utility functions (`recourse_methods.py`, `recourse_utils.py`) are adapted from the implementation of the ROAR framework.
  * *Authors:* Sohini Upadhyay, Shalmali Joshi, Himabindu Lakkaraju.
  * *Reference:* [arXiv:2102.13620 [cs.LG]](https://arxiv.org/abs/2102.13620)
  * *Repository:* [AI4LIFE-GROUP/ROAR](https://github.com/AI4LIFE-GROUP/ROAR/tree/main)

* **DualCF (Efficient Model Extraction Attack):** The conceptual framework for model extraction via counterfactuals and the evaluation of stability (Counter-Counterfactuals) is based on the DualCF methodology.
  * *Authors:* Yongjie Wang, Hangwei Qian, Chunyan Miao.
  * *Reference:* [Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency (FAccT '22)](https://doi.org/10.1145/3531146.3533188)

* **Optimal Robust Recourse with Lp-Bounded Model Change**
  * *Authors:* Phone Kyaw, Kshitij Kayastha, Shahin Jabbari
  * *Reference:* https://arxiv.org/abs/2509.21293
  *  *Repository:* [Optimal-Robust-Recourse](https://github.com/PMyatKyaw/Optimal-Robust-Recourse/blob/main/)

**Note:** The foundational recourse solvers and utility scripts in this repository are from the original ROAR implementation. This project extends those works by introducing deep model support via LIME linearization and parameter search evaluation.
This repository implements the **ROAR (Robust Recourse)** framework for generating and evaluating counterfactual explanations. It supports both simple linear classifiers and deep non-linear neural networks, using **LIME** for local linearization when dealing with non-linear networks.

---

## Quick Start

### 1. Requirements
Ensure you have Python 3.9+ installed. Install the required dependencies using:
```bash
pip install -r requirements.txt
```

### 2. Example Usages
#### deep classifier for multiple norm values

```
python param_search_main.py \
  --model-type deep \
  --norm-values 1 2 inf \
  --delta-max-values 2 \
  --lamb-values 0.2 \
  --alpha-values 0.01 \
  --size-values 1.0 \
  --epochs 100 \
  --quiet
```

#### Simple Classifier 
```
python param_search_main.py \
  --model-type simple \
  --norm-values 1 \
  --delta-max-values 0.1 0.5 \
  --lamb-values 0.1 0.5 \
  --alpha-values 0.01 \
  --epochs 50
```


### 3. Arguments Reference

| Argument | Description |
| :--- | :--- |
| `--model-type` | Model architecture to use (`simple` or `deep`). |
| `--norm-values` | Distance norms to evaluate: `1` (Manhattan), `2` (Euclidean), or `inf` . |
| `--delta-max-values` | Robustness budget ($\delta$) for the optimizer. |
| `--lamb-values` | Trade-off between validity and distance cost. |
| `--size-values` | Subsampling percentage for the query (denied) set. |
| `--epochs` | Number of training epochs for the baseline and extracted models. |

### 4.  Project Structure

| File | Description |
| :--- | :--- |
| `param_search_main.py` | **Entry point** for running grid searches and performance evaluations. |
| `recourse_methods.py` | **Core logic** for the `RobustRecourse` optimizer and weight handling. |
| `counterfactual_metrics.py` | **Generation engine** for CFs and CCFs; calculates validity and distance metrics. |
| `CombinedEvaluator.py` | **Orchestrator** of the experimental pipeline (training through extraction analysis). |
| `models/ModelWrapper.py` | **Unified interface** supporting different classifier architectures (Linear vs. Deep). |
| `recourse_utils.py` | **Utilities** for data processing and LIME explanation generation. |