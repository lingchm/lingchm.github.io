---
layout: page
title: Transformers Explained 
description: 
img: 
importance: 2
category:  
giscus_comments: true
---

# Transformers  

In deep learning, a transformer is a type of model architecture that has revolutionized the field of natural language processing (NLP) and beyond. Developed by Vaswani et al. in the paper "Attention Is All You Need" in 2017, transformers are designed to handle sequential data more effectively than previous models, like recurrent neural networks (RNNs) and long short-term memory (LSTM) networks.


## Motivation

![alt text](assets/img/blogs/transformers-three ways of processing sequences.png)

## Architecture
### Encoder-Decoder Architecture
The original transformer model consists of two main parts:
* Encoder: Processes the input sequence and generates a representation. It includes multiple layers of self-attention and feed-forward neural networks.
* Decoder: Generates the output sequence based on the encoder's representation. It also includes self-attention layers, but with an additional layer that attends to the encoder's output.

#### Self-attention

Attention is not new. We can train RNN or CNN with attention.

Transformers rely heavily on the self-attention mechanism, which allows the model to weigh the importance of different words in a sequence relative to each other. This means that each word in a sequence can attend to every other word, capturing dependencies regardless of their distance in the sequence. For example, in the sentence "The cat sat on the mat," the word "cat" can directly consider the relevance of the word "mat" through self-attention.

**Check the youtube 3 video**

#### Positional encoding
Since transformers do not have a built-in sense of the order of tokens (unlike RNNs), positional encodings are added to the input embeddings to provide information about the position of each token in the sequence. These encodings help the model understand the order of words.

There are several types of position encodings. One commonly used is sin/cos, in which nearby tokens have sinilar values, etc.
@lingchm Add other types of position encodings

#### Transformer block
* input: set of vectors x
* output: set of vectors y
* self-attention is the only interaction between vectors
* Layer norm and MLP work independent for each block

#### Other network components 
* Residual connections: Residual connections add the input of a layer to its output, helping with deeper architectures.
* Layer normalization: Transformers use layer normalization and residual connections to stabilize training and help gradients flow through the network more effectively. 
* MLP: Each layer of the transformer contains a position-wise feed-forward network that applies a non-linear transformation to each position independently. This network helps in learning complex representations.
* Tokenization is messy! trained chunking mechanism

### Diving deeper 
* what does all embeedings do?
- embedding matrix: 
- unembedding matrix: 

* How to get token-wise embeddings and input-wise embeddings?
Could 

* vocabulary matrix? 

* Why Query, Key,, Value?

* Why normalization query-wise?
 distribution

* How long can GPT models remember?
Context size. 
GPT-3 was trained with context size of 2048.
context size decides the size of t he attention matrix.
Limits how much text the GPT can incorporate/consider when making prediction for the next word

This  is why the chatbot seems to lose the thread of coonversation when long

* how does temperature control? 



## Other concepts

### Masking
There are different types of masking.
If you have a sequence of words, you can always have self-supervised learning objectives to predict for future things.

### Multi-head attention 
The self-attention mechanism is applied multiple times in parallel, called multi-head attention. Each "head" learns to focus on different aspects of the input sequence, allowing the model to capture diverse relationships and features.


### Models
![alt text](assets/img/blogs/transformer_table.png)


# Vision-Transfomers 

## Motivation
Can Attention/Transformers be used from more than text processing?
How to apply Transformer for images?

## Ideas

**Idea #1**: Standard transformer on pixels
However, too memory expensive. A R x R image needs R^4 elements per attention matrix. 

**Idea #2**: Standard transformer on patches
Works!
* Special input token: classification token 

![alt text](assets/img/blogs/transformer on patches.png)

## Discussion
ViT are (more ?) biased so need more data to perform well. As you scale the data, you get better accuracy. 

![alt text](assets/img/blogs/transformer vs resnet.png)


## Representative extensions  

### Hierarchical ViT: Swin Transformer

![alt text](assets/img/blogs/swin transformer.png)

**Window attention** 
* Motivation: 
* Solution: Rather than allowing each token to attend to all other tokens, intead divide into windows of MxM tokens (e.g. M=7), and only compute attention within each window. The total size of attention matrix is then M^4 (H/M) (W/M) = M^2HW. This is linear in image size for fixed M! 
* Problem: Tokens only interact with other tokens in the window, no interaction across windows
* Solution: Shifted Window Attention: alternate betwen normal windows and shifted windows in successive Transformer blocks
* Ugly detail: non-square windows at edges and corners


![alt text](assets/img/blogs/swin transformers shifted window.png)

@lingchm add code 



### ViLBERT: A visiolinguistic transformer

