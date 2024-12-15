---
layout: page
title: Open Vocabulary Object Detection
description: another without an image
img:
importance: 3
category: ML with limited supervision
related_publications: 
---


# OWLv2: Scaling Open Vocabulary object Detection

## Motivation

At this point:
* language has been used for object detection: ViLD, RegionCLIP
* Transformers used for detection are efficient and scalable: DETR
* OWL-ViT is a good baseline open-vocabulary transformer baed detection

In object detection, you get (i) where the object is (with a bounding box) and what it is (the class label). In contrast to image classification, where given an image, you just classify the type of image. It is a hard problem because:
* lack of labels, drawing bounding boxes takes more time
* noisy backgrounds with a lot of objects in one image
* different scales of different objects
* oclusion of objects, regression due to synthetic scenaries, etc.  

Open vocabulary object detection has wide applications in robotics, autonomous driving. Try out yourself at hugging face xx. 

The main goal of this work is to scale training for OWL-ViT to close the scarcity gap to improve the performance for the long-tail. The new methodology for training is called OWL-ST and the new model is called OWLv2. The method is based on self-training. 


## Related Works

### OWL-ViT Architecture
* Text transformer encoder 
* Vision transformer encoder
* The image embedding and text embedding are "combined" through contrastive loss over images in a batch
* Text/vision encoder initialized from CLIP or SigLIP
* Training recipe:
    * contrastively pre-train image nad text encoders on large-scale image-text data
    * add detection heads and fine-tune on medium-sized detection date
* Engineering efforts focused on generalization such as augmentation and regularization
* OWL-ViT-L/14 trained on 2M samples 

### ViLD (2021)
* Vision and Langauge knowledge distillation
* Teacher: encodes category text and image regions of object proposals
* Student (two-stage detector) whose region embeddings of detected boxes are aligned wit hthe text and image embeddings inferred by the teacher

### RegionCLIP (CVPR 2022)
* Used N-gram
* 

### DETR (ECCV 2020)
* Detection Transformer
* Removed the decoder and just use the outputs from the encoder
* Bipartite matching loss is adjusted for open-vocabulary detection

Q: Why removing the encoder is helpful?
* In encoder-decoder, you have more flexibility, gives you more parameters and you can determine the number of outputs from the decoder
* In encoder-only, the number of input patches is the number of output embeddings. So it is less flexible but simpler.

Q: Are positional embeddings learnable params?
* No, it is some function of (x,y).
* With the positional embeddings, we could shuffle the input tokens and still get the same results.

Q: Past works only used nouns from sentences as classes?
* In RegionCLIP, they had a concept pool
* If using N-gram caption, it would be more comprehensive


## Method

Overview
1. Generate pseudo-box annotations on WebLI with OWL-ViT L/14 queried with caption N-grams
2. Train new models on pseudo-annotations
3. Optionally, fine-tune on human annotations.

The main contribution of this paper is the scaling up on OWL-ViT L/14. This was by training on WebLI
* 10B image-text pairs from internet 
* Data in 109 languages, automatically translates non-English text
* Dedups pairs from 68 common datasets, 0.38% shrinkage
* Label space: set of all possible labels that can be assigned 
    * Human-annotated
    * Machine-generated label space:
        * N-grams: generate n number of sequence tokens for the same object
        * Parsed captions for all word N-grams up to length 10
        * Filters pseudo-labels by empirically determined threshold (0.1)

**Self-training** on OWL-ViT L/14 by:
1. Given an image, the model detects objects in the image in a class-agnostic way. Keep as long as the model is remotely confident with the "objectness" score.
2. Given list of free-text queries, the model produces scores indicating the likelihood that each detected object is associated with each text query.

Self-training is much more efficient than hand-annotator in terms of scaling.



**Other engineering ideas**
* Token dropping: 
    * Problem: less informative tokens are inefficient
    * Solution: drop 50% of tokens lower than mean pixel variance
* Instance selection:
    * Problem: each output toekn from encoder predicts bbox, most of which won't be objects
    * Solution: "objectness heads" to predict likelihood an output token represents an object
* Mosaics
    * Problem: ODiW/real-world samples more complex than WebLI training data
    * Solution: combine multiple samples into grid-like samples (e.g. 1x1, 4x4, 6x6)
    * A caveat is that the token dropping is still applied to mosaic images so not really dropping contents from the image but the empty space between

Results show that fine-tuning on human-curated dataset yields
* +4.3 from B/15 and +1.3 from L/14 on xx
* +0.3 from B/15 and -1.4 -> less generalizable and more specific

In most cases, the combination of self-training and fine-tuning works best. In some cases, using curated vocab has better performance and sometimes not. Trade-off between generalizability and specificity. 
* Longer self-trainng improves open-world performance
* Longer fine-tuning improves dataset specific performance with decreases in open-world performance
* Weight ensembling balances tradeoffs 

(image comparing fine-tuned vs not fine-tuned from ppt)


Q: What are disadvantages of self-training?
* Can create a negative feedback loop
* Depends on quality of pseudo-labels and noise of the data
* Similar to knowledge distillation

Q: How big the vocabulary is?
* N-grams with N=10 is a lot


## Discussion

Limitations
* built on CLIP with has broader impacts and potential concerns (CLIP fine-grained detection is "near-random")

Strengths
* self-training enables the study of large scale open-vocab object detection
* Signficant improvements in detection performance using weak supervision

Weaknesses
* Despite scaling training from ~20M to 10B, barely beats GLIPv2 in ODiW
* SigCLIP is released after CLIP with better accuracy and efficiency, yet only used for one model
* Sensitivity to pseudo-annotations filtering threshold 

 






