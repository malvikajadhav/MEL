📌 [View Poster (PDF)](./poster.pdf)

# 🧪 Poisoning Attacks on Multimodal Entity Linking Models

This project explores how **data poisoning attacks** can undermine the performance and reliability of **Multimodal Entity Linking (MEL)** systems. MEL aims to associate entity mentions in text, images, and videos with corresponding entries in a structured knowledge base. However, adversarial actors can inject poisoned data into training sets to manipulate or mislead model predictions.

## 📄 Paper

You can read the full research paper here:  
[🔗 Poisoning Attacks on Multimodal Entity Linking Models (PDF)](https://zenodo.org/records/15844783)

## 📚 Datasets

- **[WikiDiverse (Wang et al., 2022)]**: 7,824 image-text pairs, 16,327 mentions.
- **[WikiMEL (Luo et al., 2023)]**: 22,136 image-text pairs, 25,846 mentions.

Both datasets were split into training, validation, and test sets, following an 8:1:1 and 7:1:2 ratio respectively.

## 🏗️ Model Architecture: GEMEL

The GEMEL (Generative Multimodal Entity Linking) model architecture consists of three core components:

1. **Vision-Language Alignment**  
   Projects image features into the text embedding space using a frozen image encoder and a trainable visual prefix mapper.

2. **In-Context Learning (ICL)**  
   Provides multimodal demonstrations to guide the LLM in generating accurate entity predictions.

3. **Constrained Decoding**  
   Uses a prefix tree of valid entity names to restrict output to knowledge base entries.

## 🧪 Poisoning Techniques

We apply **textual data poisoning** using:

### 1. [TextAttack (Morris et al., 2020)](https://arxiv.org/abs/2005.06620)
- *Embedding*: Synonym replacement without changing sentence embeddings.
- *WordNet*: Semantic swaps using WordNet synonyms.
- *RandomSwap*: Character-level perturbations.

### 2. [CLARE (Li et al., 2020)](https://arxiv.org/abs/2009.07502)
- Context-aware infilling using Replace, Insert, and Merge operations.
- Experiments conducted with poisoning rates: 10%, 20%, 30%, and 40%.

## 🛠️ Infrastructure

- Trained on **HiPerGator (UF HPC)** with A100 GPUs.
- Sentence embeddings generated using **SimCSE**.
- Poisoning generated using **TextAttack** and **CLARE** pipelines.

## 👥 Authors

- [Malvika Jadhav](mailto:jadhav.m@ufl.edu)
- [Sam Maley](mailto:smaley@ufl.edu)

## 📄 Citation

If you use or build on this work, please cite:

```bibtex
@misc{jadhav2025mel,
  author       = {Malvika Jadhav and Samuel Maley},
  title        = {Multimodal Entity Linking under Adversarial Settings},
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15844783},
  url          = {https://doi.org/10.5281/zenodo.15844783}
}
