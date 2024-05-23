# VrDU-Doctor 
**We research why and how VrDU Multimodal models answer what they answer**

## Introduction
Visually-rich Document Understanding (VrDU) consists of a Deep Learning (DL) Model synthesizing or selecting information from documents (images with text) to answer a question or classify a chunk of text. VrDU tasks are multimodal, i.e., models use information from text, images, or even the document layout to solve the tasks.

Humans have worked with different formats of documents since the beginning of History (inscriptions, cards, books, etc.). Nowadays, we still work with non-digital documents: we still receive a medical record when visiting the doctor or applying to the university with a transcript of records. On the other hand, our lives are increasingly digital: part of our relevant data is digital; therefore, we deal with data transfer from the analogical to the digital domain.

There are multiple examples:
+ :ticket: Purchase tickets :arrow_right: Splitwise/Tricount :dollar::calling:
+ :bookmark_tabs: Transcript of records :arrow_right: University database :100: :bar_chart:
+ :scroll: Your rental contract :arrow_right: Tax Agency :cop::moneybag:

Recent contributions have shown the potential models that can automate data gathering and bridge the gap between the analogical and digital domains. Nevertheless, they have not paid much attention to answering some questions.

+ **What are the contributions of the multimodal parts?**
+ **Could we avoid using any modal part?**
+ **Is every piece of data equally important to solve a well-defined task?**
+ **What is the perfect dataset size to fine-tune a VrDU model?**
+ **Could we achieve similar results, improving the quality and reducing the amount of pre-training data?**


**Let's dive in** :dolphin:

## Models
We are researching with one of the SOTA family models: LayoutLM. We expect to broaden our scope soon:

- [x] LayoutLMv2
- [x] LayoutXLM
- [x] LayoutLMv3
- [ ] Donut
- [ ] Udop

## Dataset
We use the OOL Dataset (A Cool Tool for School Fool). The OOL Dataset is a synthetic multimodal dataset (Image + Text + Layout) crafted explicitly for the Visually-rich Document Understanding task. The OOL Dataset contains 33k fully labeled samples and gathers students' records in English and Spanish. You can find more about the OOL Dataset here:

+ **Paper:** Under development :hammer_and_wrench:
+ **Code:** [OOL Dataset](https://github.com/nachoDRT/OOL-Dataset)
+ **Dataset:** Under development :hammer_and_wrench:
+ **Benchmark:** [OOL Dataset Benchmark](https://wandb.ai/iderodrigo/OOL_Dataset)

## Results

Under development :hammer_and_wrench:

## Team

We are researchers from **Comillas Pontifical University**
 - **Ignacio de Rodrigo [@nachoDRT](https://github.com/nachoDRT)**: PhD Student. Benchmark Design, Software Development, and Data Analysis.
 - **Alberto Sánchez [@ascuadrado](https://github.com/ascuadrado)**: Research Assistant. Benchmark Design and  Data Analysis.

## Citation
If you find our research interesting, please cite our work. :page_with_curl::black_nib: