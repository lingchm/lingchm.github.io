---
layout: page
title: Eval datasets
description: a project with a background image
img: assets/img/1.jpg
importance: 3
category: data mining
---


Why multimodality matter?
* Faithfulnes: human experience is multimodal 
..
* language is a universal way to describe. Improve generalization of vision-based scene understanding via language. 


## Tasks

**Open-Vocabulary classification and Detection**
* Types:
    *  Image + vocab -> label
    * Image + vocab -> label + bbox
* Limitations:
    * Still bounded to the training or inference data 
    * Bounded to the captions
* Potential improvements to generalize?
    * Could we use the open-world dictionary of all possible nouns?
        * Search space is too large
        * Nouns that we have in images or conversations is a biased set of them. If we evaluate on images, the zero-shot performance would go down...

**(Generalized) Referring Expression**
* Idea: describe anything and have it be detected!
* (CHECK)

**Image captioning**
* Image -> text
* an early vision-language task
* Captions can vary in detail/how fine-grained it is
* Metrics
    * BLUE: quantify the quality of machine-generated outputs (precision-focused)
    * ROUGE (recall-focused): what percentage of the words or n-grams in the reference occur in the generated output
    * Perplexity: confidence of predicting next token
    * METEOR: introduced in semantic matching, based on how well the generated sentences are aligned
    * CIDEr: recently introduced for image captioning task
    * MRR
    * BERTSCore
    * Human evaluation 

**Open-Ended Object Detection**
* Image -> bbox -> text
* Some papers attempt to combine text generation and detection

**Image Generation**
* Text -> image
* Language is used to condition multi-modal generation 
* Can be generalized to images, videos, audio, etc. 
* Models: Dyffusion

**Visual Question answering**
* What does "understanding" mean? 
* There are various ways to investigate. Classification, detection, generation are some ways. Visual question-answering is a helpful way
* VQA is a new dataset containing open-ended questions about images. There is typically at least 3 questions per image and 10 ground truth answers per qusetions with 3 plausible (but incorrect) answers per question. 

Issues
* It turns out that for many questions, vision is not necessary?
    * There are common sense knowledge that has already been embedded during training
    * Language can have a strong prior. Some models are not really forcing the usage of vision.
    * e.g. "Do you see a ...", a blind answer of "yes" will get VQA accuracy of 87%.
    * Strategies:
        * modality dropout
        * add terms to the loss
        * fix the dataset 
        * plot distributions to make sure dataset is not biased 

**VQAv2**


**Other forms**
- optical character recognition
- documents/infographic understanding
- keypoint detection
- video/action recognition
- cross-image alignment

**Decision-making**
* image + text -> action
* examples:
    * image -> math reasoning
    * image -> code
    * image -> action 

## Datasets
* Typically, there are variaous pre-training and finetuning datasets. Then, evaluation is done either zero-shot or finetuned on validation sets
* Some datasets have variants, euch as Ref/gRefCOCO
* Out-of-distribution variants
    * distribution shifts to images: IV-VQA, CV-VQA
    * distribution shifts to questions: VQA-rephrasing
    * distribution shifts to answers
    * distribution shifklts to multi-modalities


## Discussion
* what 
