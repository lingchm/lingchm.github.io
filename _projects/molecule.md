---
layout: page
title: Analyzing Steered Molecular Dynamics
description: AI for understanding molecular dynamics
img: assets/img/projects/tcr_cd3.png
importance: 4
category: multimodal
---

In the context of structural biology, an active area of research is to understand molecular processes relevant to cancer research, particularly interactions between the T cell receptor (TCR) and Programmed Death 1 (PD-1) with their respective ligands: antigen peptide-major histocompatibility complex (pMHC) and PD-Ligands. These binding processes are critical to the immune system’s ability to identify and destroy cancer cells, forming the basis of adoptive T cell therapy. 

**Molecular Dynamics** (MD) simulations offer a powerful tool for investigating these complex biophysical systems at an atomic level, revealing force-regulated and time-dependent molecular interactions, such as receptor-ligand dissociation. However, MD simulations generate highly complex, high-dimensional data, with tens of thousands of atoms tracked at nanosecond resolution. Current analytical methods often rely on predefined measures, such as global-level or domain-level distances, which can introduce human bias and may overlook essential properties in the simulation. Additionally, in steered MD experiments, where force is applied to trigger conformational changes to study functional differences among compounds, data is often presented as a mixture of random atomic noise and force effects, making it challenging to observe the critical local interactions.

In this project, our goal is to develop an AI framework to analyze MD data on conformational changes and receptor–ligand dissociation. Key research questions include:
* How to disentangle global movements induced by force as well as local interactions 
* Identify key residues responsible for the conformational change
* Modelling the high-resolution temporal dynamics

<!-- We approach this as a dynamic graph representation learning problem across two key time points: before and after force application. Since molecules are inherently graph structures, we employ Graph Neural Networks (GNN) to learn latent node-level representations. In the first stage, our model disentangles force-invariant features from force-induced changes using contrastive learning techniques. In the second stage, we further decompose the force-induced changes into global and local movements through self-supervised tasks, such as link prediction. To ensure the meaningfulness of these representations, constraints are imposed on the global movements (e.g. low-rank or linearity along the pulling direction) and local movements (e.g. sparsity). This disentangled dynamic graph representation approach allows biologists to gain insights into both global molecular adaptations to external constraints as well as local interactions within specific domains or regions, offering a more interpretable and comprehensive analysis of high-dimensional MD data for understanding of functional differences across different TCR complexes. -->

*This project is in collaboration with the [Cellular and Molecular Biomechanics Lab](https://thezhulab.github.io/zhu_lab_website/) lead by Dr. Cheng Zhu from the Wallace H. Coulter Department of Biomedical Engineering*.