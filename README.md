<div align="center">

# DiFRa: A Unified Framework for Harmonizing Semantic Diversity and Factual Consistency in Question-Answer Generation

[![Paper](https://img.shields.io/badge/Paper-ACL_Anthology-firebrick.svg)](https://github.com/zqinli/DiFRa)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

Question-Answer Generation (QAG) is essential for alleviating the cold-start problem in domain-specific large language model (LLM) post-training, where high-quality data is severely scarce. Effective training samples include rich semantic diversity and rigorous factual consistency. Thus, it is necessary to consider the inherent tension between semantic breadth and factual fidelity. However, it is extremely challenging to trade off semantic diversity against factual consistency, in that generalization across the semantic space must be achieved effectively and reliably, and factual integrity must be ensured as well. To address this issue, we propose an effective framework, namely DiFRa, that integrates continuous concept diffusion with discrete knowledge graph constraints to balance semantic diversity and factual consistency. Specifically, the proposed DiFRa models discrete concepts as a continuous latent distribution to sample embeddings that capture rich semantic variations, and constructs a refined knowledge graph as explicit factual constraints. Then, a diversity and consistency aware mechanism is designed to dynamically integrate both embeddings and the knowledge graph for QA pairs generation. Furthermore, we introduce SeFa, which harmonizes semantic entropy and consistency scores to quantify the trade-off between diversity and correctness. Extensive experiments demonstrate that DiFRa consistently outperforms the baseline models, validating its efficacy in reconciling the tension to generate semantically diverse and factually consistent QA pairs. 

