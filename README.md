# VERSE 📄🔍👀
### Visual Embedding Reduction and Space Exploration — Clustering-guided Insights for Training Data Enhancement in V-rDu

<p align="center" style="margin-top: 50px; margin-bottom: 50px;">
  <img src="imgs/verse.png" alt="verse" width="200" /><br>
</p>


## Introduction :information_source:
Visually-rich Document Understanding (VrDU) consists of a Deep Learning (DL) Model synthesizing or selecting information from documents (images with text) to answer a question or classify a chunk of text. VrDU tasks are multimodal, i.e., models use information from text, images, or even the document layout to solve the tasks.

Humans have worked with different formats of documents since the beginning of History (inscriptions, cards, books, etc.). Nowadays, we still work with non-digital documents: we still receive a medical record when visiting the doctor or applying to the university with a transcript of records. On the other hand, our lives are increasingly digital: part of our relevant data is digital; therefore, we deal with data transfer from the analogical to the digital domain.

<p align="center" style="margin-top: 50px; margin-bottom: 50px;">
  <img src="imgs/introduction.png" alt="Introduction" width="600" /><br>
  <em> Figure 1. Traditionally, the quality of synthetic images in a dataset is assessed from an anthropocentric perspective, answering the question of whether such images appear photorealistic. In contrast, this work proposes evaluating the images from the model’s perspective through visual embedding analysis.
  </em>
</p>

## Models (TO-DO)
We are researching with one of the SOTA family models: LayoutLM. We expect to broaden our scope soon:

- [x] LayoutLMv2
- [x] LayoutXLM
- [x] LayoutLMv3
- [ ] Donut
- [ ] Idefics2
- [ ] PaliGemma
- [ ] LLaVA

## Datasets 📑

### Training Dataset 📄
We use the Spanish partition of the MERIT Dataset. The MERIT Dataset is a synthetic multimodal dataset (Image + Text + Layout) crafted for Visually-rich Document Understanding tasks. You can find more about MERIT Dataset here:

+ **Dataset:** [MERIT Dataset @ Hugging Face 🤗](https://huggingface.co/datasets/de-Rodrigo/merit)
+ **MERIT Dataset Paper:** [@ Pattern Recognition](https://www.sciencedirect.com/science/article/pii/S0031320325011653) and [@ ArXiv](https://arxiv.org/abs/2409.00447)
+ **Pipeline code:** [MERIT Dataset generation pipeline](https://github.com/nachoDRT/MERIT-Dataset)

<p align="center" style="margin-top: 50px; margin-bottom: 50px;">
  <img src="imgs/dataset_overview.png" alt="Dataset overview" width="600" /><br>
  <em>
    Figure 2. Training samples used. We employ the Spanish-language subsets of the MERIT Dataset, across its different versions (A). Each version comprises data from seven different schools (B). New versions complement the vanilla MERIT Dataset, composed of digital document samples (C) and their renderized versions (D).  More information available in the
    <a href="https://www.sciencedirect.com/science/article/pii/S0031320325011653">MERIT Dataset paper</a>.
  </em>
</p>

### Test-Dev Dataset 📜

We use MERIT Secret (a real dataset under Non-Disclosure Agreement) as test-dev dataset.

## Methodology 🔄

<p align="center" style="margin-top: 50px; margin-bottom: 50px;">
  <img src="imgs/methodology.png" alt="Methodology" width="600" /><br>
  <em>
    Figure 3. VERSE methodology. More information available in the
    <a href="https://arxiv.org/">VERSE paper</a>.
  </em>
</p>


## Results 📈

WIP :hammer_and_wrench:

<p align="center" style="margin-top: 50px; margin-bottom: 50px;">
  <img src="imgs/embeddings_animation_PCA_Donut.gif" alt="embeddings animation Donut" width="400" /><br>
  <em>
    Figure 4. Synthetic training-samples moving across the Reduced Embedding Space (RES) of Donut. Every step shows the same sample under increasing level of visual information (purple). In background, PC maps showing F1 scores of the target (test-dev) samples.
</em>
</p>



## Team

We are researchers from **[Comillas Pontifical University](https://www.iit.comillas.edu/)**
 - **Ignacio de Rodrigo [@nachoDRT](https://github.com/nachoDRT)**: PhD Student.
 - **Álvaro López [@allopez](https://www.iit.comillas.edu/personas/allopez)**: Supervisor.
 - **Jaime Boal [@jboal](https://github.com/jboalml)**: Supervisor.

## Citation
If you find our research interesting, please cite our works. :page_with_curl::black_nib:

**VERSE**
```bibtex
@article{WIP,
  title={WIP},
  author={WIP},
  journal={arXiv preprint arXiv:WIP},
  year={2025}
}
```

**MERIT Dataset**
```bibtex
@article{deRodrigo2025merit,
title = {The MERIT dataset: Modelling and efficiently rendering interpretable transcripts},
journal = {Pattern Recognition},
volume = {172},
pages = {112502},
year = {2026},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2025.112502},
url = {https://www.sciencedirect.com/science/article/pii/S0031320325011653},
author = {Ignacio {de Rodrigo} and Alberto Sanchez-Cuadrado and Jaime Boal and Alvaro J. Lopez-Lopez},
```