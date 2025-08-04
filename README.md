<div align="center">
<h1>ProLearn: Alleviating Textual Reliance in Medical Language-guided Segmentation via Prototype-driven Semantic Approximation</h1>

<a href="https://arxiv.org/pdf/2507.11055" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Paper-ProLearn" alt="Paper PDF">
</a>
<a href="https://arxiv.org/abs/2507.11055"><img src="https://img.shields.io/badge/arXiv-2507.11055-b31b1b.svg" alt="arXiv"></a>
<a href="https://shuchangye-bib.github.io/websites/ProLearn/prolearn.html"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>

**[The University of Sydney](https://www.sydney.edu.au)**; **[Macquarie University](https://www.mq.edu.au)**

[Shuchang Ye](https://shuchangye-bib.github.io), [Usman Naseem](https://researchers.mq.edu.au/en/persons/usman-naseem), [Mingyuan Meng](https://mungomeng.github.io), [Jinman Kim](https://www.sydney.edu.au/engineering/about/our-people/academic-staff/jinman-kim.html)
</div>

## 🚀 Overview

**ProLearn** introduces a significant advancement beyond our previous work, [SGSeg](https://github.com/ShuchangYe-bib/SGSeg), by deeper analyzing and addressing one of the core limitations of medical language-guided segmentation: *textual reliance*.

🔍 Why is textual reliance a problem?

📝 Most medical segmentation datasets lack paired reports, leaving large amounts of image-only data unused for training.

📝 Inference often requires text input, which is impractical in real clinical workflows, where segmentation usually precedes reporting.

🧠 **ProLearn**: the first prototype-driven learning framework that enables 1) image-only, image-text data mix training; 2) inference with limited or no textual input.

<p align="center">
  <img src="static/images/visualization.svg" width="85%">
</p>

---

## 📉 Performance Under Limited Text

To simulate real-world incomplete pairing, we train ProLearn with only **1% to 50%** paired text data and compare it with SOTA language-guided models. Unlike others, **ProLearn maintains performance even under extreme text scarcity**.

<p align="center">
  <img src="static/images/degradation.svg" width="75%">
</p>

---

## 🔬 Qualitative & Interpretability Analysis

ProLearn produces robust and localized segmentation maps, even without text. Its **PSA module** preserves attention saliency and lesion coherence — outperforming baselines like SGSeg and LViT.

<p align="center">
  <img src="static/images/visualization.svg" width="80%">
</p>

---

## Quick Start

First, clone this repository to your local machine and install the dependencies. 

```bash
git clone git@github.com:ShuchangYe-bib/ProLearn.git
cd ProLearn
conda create --name prolearn python=3.11
conda activate prolearn
pip install -r requirements.txt
```

---

## 📚 Citation

If you find ProLearn useful in your research, please consider citing:

```bibtex
@misc{ye2025prolearn,
  title={Alleviating Textual Reliance in Medical Language-guided Segmentation via Prototype-driven Semantic Approximation},
  author={Shuchang Ye and Usman Naseem and Mingyuan Meng and Jinman Kim},
  year={2025},
  eprint={2507.11055},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2507.11055}
}
```
