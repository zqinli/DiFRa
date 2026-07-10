<div align="center">

# DiFRa: A Unified Framework for Harmonizing Semantic Diversity and Factual Consistency in Question-Answer Generation

</div>

[![Paper](https://img.shields.io/badge/Paper-ACL_Anthology-firebrick.svg)](https://aclanthology.org/2026.findings-acl.1493)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



## 📌 Overview

**DiFRa** is a unified framework for **Question-Answer Generation (QAG)**. It is designed for alleviating the cold-start problem in  domain-specific LLM post-training scenarios where high-quality QA data is scarce.

High-quality generated QA pairs should satisfy two requirements at the same time:

- **Semantic Diversity**: generated QA pairs should cover rich and varied semantic information instead of simply copying source spans.
- **Factual Consistency**: generated QA pairs should remain faithful to the original context and avoid hallucinated or contradictory facts.

However, these two goals are naturally in tension: stronger semantic generalization can introduce factual errors, while overly conservative generation often lacks diversity. DiFRa addresses this challenge by combining **continuous concept diffusion** with **discrete knowledge graph constraints**.

<p align="center">
  <img src="assets/challenges.svg" width="66%" alt="Motivation of DiFRa">
</p>
<p align="center">
  <em>Figure 1: Challenges. DiFRa aims to generate QA pairs that are both semantically diverse and factually consistent.</em>
</p>

---

## ✨ Key Features

- **Concept Construction and Diffusion (CCD)**  
  Extracts topics and key phrases from the source context, then maps discrete concepts into a continuous latent space. A diffusion model samples diverse concept embeddings to encourage semantic generalization.

- **Factual Constraint Construction (FCC)**  
  Builds and refines a knowledge graph from the source context. The refined graph provides explicit factual constraints for QA generation.

- **Diversity and Consistency Aware Mechanism (DCAM)**  
  Dynamically integrates context embeddings, diffused concept embeddings, and refined knowledge graph embeddings into a frozen LLM to guide QA generation.

- **SeFa Evaluation Metric**  
  Introduces **SeFa**, a harmonic-style metric that combines **Semantic Entropy (SE)** and **Factual Consistency (FC)** to measure the balance between diversity and correctness.

---

## 🧠 Framework

<p align="center">
  <img src="assets/overview.svg" width="100%" alt="Overall architecture of DiFRa">
</p>



<p align="center">
  <em>Figure 2: Overall architecture of DiFRa. The framework contains Concept Construction and Diffusion, Factual Constraint Construction, and the Diversity and Consistency Aware Mechanism.</em>
</p>
Given an input context, DiFRa follows three main steps:

1. **Concept Construction and Diffusion**  
   DiFRa extracts topic words and key phrases to form a concept set, then performs conditional diffusion in the continuous embedding space to obtain diverse concept embeddings.
2. **Factual Constraint Construction**  
   DiFRa constructs a fine-grained knowledge graph from the context and refines it into a compact factual graph that retains essential facts while removing redundant or fragmented triples.
3. **Diversity and Consistency Aware Generation**  
   DiFRa projects the diffused concept embeddings into the LLM embedding space and concatenates them with context and knowledge graph embeddings. The frozen LLM then generates QA pairs under both semantic and factual guidance.

---

## 🚀 Installation

```bash
git clone https://github.com/zqinli/DiFRa.git
cd DiFRa

conda create -n difra python=3.12 -y
conda activate difra

cd iworkplace
pip install -e ".[all]" --index-url https://pypi.tuna.tsinghua.edu.cn/simple

bash examples/scripts/run.sh
```
---
## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{li-etal-2026-difra,
    title = "{D}i{FR}a: A Unified Framework for Harmonizing Semantic Diversity and Factual Consistency in Question-Answer Generation",
    author = "Li, Zhenqin  and
      Ding, ShengYong  and
      Li, Shuangyin",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.1493/",
    doi = "10.18653/v1/2026.findings-acl.1493",
    pages = "29857--29875",
    ISBN = "979-8-89176-395-1"
}
```


## 🙏 Acknowledgement

This project is partially inspired by the open-source project [DiffuSeq](https://github.com/Shark-NLP/DiffuSeq). We thank the authors for their valuable contribution.

---

## ⭐ Star History

<a href="https://www.star-history.com/?repos=zqinli%2FDiFRa&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=zqinli/DiFRa&type=date&theme=dark&legend=top-left&sealed_token=hY7WC7os2rlCe3lP3l3spPDNN_ZqV55NN1Ywjw0wZiJR6gDqXoOcxA8pInqR9qlVF0foTHBDzGC2KEOia83aU-YlULwON1Kp3ji-rtMndYkULiy66uc6Pg" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=zqinli/DiFRa&type=date&legend=top-left&sealed_token=hY7WC7os2rlCe3lP3l3spPDNN_ZqV55NN1Ywjw0wZiJR6gDqXoOcxA8pInqR9qlVF0foTHBDzGC2KEOia83aU-YlULwON1Kp3ji-rtMndYkULiy66uc6Pg" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=zqinli/DiFRa&type=date&legend=top-left&sealed_token=hY7WC7os2rlCe3lP3l3spPDNN_ZqV55NN1Ywjw0wZiJR6gDqXoOcxA8pInqR9qlVF0foTHBDzGC2KEOia83aU-YlULwON1Kp3ji-rtMndYkULiy66uc6Pg" />
 </picture>
</a>