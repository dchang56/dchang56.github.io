---
published: true
title: Review | Deep Learning for Healthcare - review, opportunities, and challenges
use_math: true
category: Literature Review - DLEHR
layout: default
---

# Table of Contents

* TOC
{:toc}


# Authors

Riccardo Miotto (Mount Sinai); Fei Wang (Weill Cornell Medicine)

# Introduction

Key challenge: getting knowledge and actionable insights from complex, high-dimensional, heterogeneous biomedical data.

growing availability of data; precision medicine; 

high-dimensionality, heterogeneity, temporal dependency, sparsity, irregularity [13-15]

ontologies and inconsistencies: SNOMED-CT, UMLS, ICD-9 [16-19]

requirement of domain expert/manual feature engineering: no scalability and very limited

Representation learnin: minimizes manual intervention and domain knowledge requirements, uses raw data [21,22]


DeepMind is diving into health care [28]
(other big DL companies for healthcare)


**Clinical imaging**

Followed success in computer vision; first application of DL to clinical data 

**EHR**

structured (diagnosis, medicatinos, lab tests, etc)
unstructured (free text clinical notes)

usually specific, supervised predictive clinical task
common: DL method that outperforms conventional models wrt certain performance metrics (AUROC, accuracy, F-score)
some also use unsupervised methods to get latent patient representations

some examples: DeepCare, Doctor AI, Deep Patient (others on the table)
time series: Lipton, Che, Lasko, Razavian
neural language models: Tran,Nguyen (Deepr)

**Challenges and opportunities**

1. Data volume: need a lot of data to train complex networks (why CV, NLP are so successful)
2. Data quality: highly heterogeneous, ambiguous, noisy, incomoplete (sparsity, redundancy, missing values)
3. Temporality: diseases and patients change over time; many models assume statis vector-based inputs; can't handle time well
4. Domain complexity: phenomena trying to study are very complex and not completely known
5. Interpretability: black boxes. reason why algorithms works is important for convincing professionals about recommendations and results

Directions:
1. Feature enrichment: use lots of features 
2. Temporal modeling (RNN, attention, memory)
3. Interpretable modeling

scale to billions
unified patient representation
