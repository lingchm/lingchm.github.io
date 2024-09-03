---
layout: page
title: Discovery of Biomarkers for Post-traumetic Headache
description: a project that redirects to another website
img: assets/img/7.jpg
redirect: https://unsplash.com
importance: 3
category: multi-modal learning
---


# Vision-Language Models

## Introduction
![alt text](assets/img/blogs/survey vlm.png)

Hard to debug in general

## Architecture Overview
Three main components:
1. modality encoder
2. Large-language model
3. Modality interface

## Modality encoders
* common image encoders:
 * Self-supervised: DINO, MAE
 * CLIP
 * EVA-CLIP

* Considerations
 * key question of tokenization: beyond just classificaiton, need finer-grained
 * one SSL objective does not learn everything. One objective may learn spatial information, some semantics, etc. So people try to concatenate everything
 * Higher resolution inputs improve performance. Multi-resolution methods


## Large-language models
* Popular options
 * Flan-T5
 * LlaMA
 * Vicuna
 * Wen

* Considerations
 * Free vs cost
 * Larger tend to perform better


## Modality-interfaces

* Token-level fusion: these methods aim to create a unified representation that captures the interactions between vision and language tokens. "tokens" refer to discrete units of information from each modality. For language, tokens are typically words or subwords, while for vision, tokens are often features extracted from image patches or regions. There are multiple ways to combine language and vision tokens.
    * Cross-Attention Mechanisms allows the model to align and integrate visual and textual tokens by attending to one modality's tokens while considering the other modality’s tokens.
        * Visual Tokens Attend to Textual Tokens: When processing visual tokens (e.g., image patches), the model can use attention mechanisms to focus on relevant textual tokens (e.g., words in a caption).
        * Textual Tokens Attend to Visual Tokens: Conversely, textual tokens can attend to visual tokens to derive context or meaning related to specific visual features.
    * Token Alignment. Token alignment involves mapping visual tokens to textual tokens and vice versa. This alignment ensures that each visual token corresponds to a relevant textual token and helps in learning how visual features relate to textual descriptions. Techniques include:
        * Bilinear Pooling: Combining features from visual and textual tokens through bilinear pooling to capture interactions.
        * Attention Maps: Using attention maps to align and fuse features from both modalities
    * Joint Embedding Spaces: Token-level fusion can be achieved by projecting both visual and textual tokens into a shared embedding space. This shared space enables direct comparison and interaction between visual and textual tokens. 
        * Multimodal Transformers: Transformers that process both visual and textual tokens in a unified architecture, learning joint representations through self-attention and cross-attention layers.
    * Feature Concatenation. Visual and textual tokens can be concatenated or combined into a single representation before further processing. This approach integrates features from both modalities at each token level:
        * Concatenation: Directly concatenating features from visual and textual tokens and feeding them into subsequent layers.
        * Sum or Average Fusion: Summing or averaging features from both modalities to create a combined representation.
    * Linear projection (LlaVa)
    * Query-based like Q-Former (BLIP-2)

* Feature-level fusion:
    * Cross-attention layers (Flamingo)
    * Visual expert modules (CogVLM)

* Comments
    * Connection modules typically account for small percentage of parameters 

## Training stages
1. Pre-trainng of vision and language models
    * Collect examples of (instruction, output) pairs across many tasks and finetune an LM, then evaluate on unseen tasks
    * Goal: align modalities and learn multimodal knowledge.
    * Data; mostly large volumes of image-text pairs. 
    * Datasets: CC3M, LAION-5B, COYO-700M, or GPT-4V for generating fine-grained data
    * why: Need to do to avoid biases/hallucination from data and LM
2. Instruction tuning 
    * Goal: teach models to follow multimodal instructions
    * Data collection methods:
        * Adapting existing datasets
        * Self-instruction: use a LLM to expand the instructions
        * Mixing language-only and multi-modal data
    * LlaVa-instruct: bounding boxes/captions -> GPT4 -> more data
    * Data quality is important
3. Alignment tuning
    * Goal: align outputs with human preferences
    * Reinforcement Learning with Human Feedback (RLHF)
    * Direct Preference Optimization (DPO)
    * Key papers:
4. Prompting
    * Zero-shot prompting is common
    * In-context evalaution is an interesting area
        * less studied in multi-modal space 
5. Evaluation methods
    * Closed-set: task-specific datasets and metrics
        - MME/MMBench
        - Video-ChatGPT, Video-bench
    * Open-set:
        - mPLUG-Owl: visually related evaluations, diagrams, flowcharts, etc.
    * Autoevaluation common: GPT scoring
