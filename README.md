# AI Text Detectability Analysis
All 1000 rows of data is available in `dataset.csv` with 119 rows having NaN values resulting in a total usable row count of 881. Claude Sonnet 3.5 was used to generate portions of the code as well as clean it. 

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Key Findings](#key-findings)
5. [Detailed Analysis](#detailed-analysis)
6. [Visualizations](#visualizations)
7. [Conclusions](#conclusions)
8. [Future Work](#future-work)

## Introduction

This project aims to analyze the factors that contribute to making AI-generated text less detectable by AI detection systems. We employ a variety of statistical and deep learning techniques to understand the linguistic features, semantic properties, and structural elements that influence the "human-likeness" of text.

## Dataset

Our analysis is based on a dataset containing:
- 881 text samples
- Each sample includes an original text and its "undetectable" version
- Associated AI detectability scores for both versions

## Methodology

We employed a multi-faceted approach to analyze the text data:

1. **Feature Extraction**: We extracted a wide range of linguistic features including:
   - Part-of-speech (POS) tags
   - Named Entity Recognition (NER) tags
   - Readability scores
   - Syntactic complexity metrics
   - Paraphrasing features

2. **Deep Learning Models**:
   - Attention-based Neural Network
   - Feature Interaction Network
   - Multi-Task Learning Network

3. **BERT-based Semantic Analysis**: We used BERT embeddings to analyze semantic similarities between original and undetectable texts.

4. **Statistical Analysis**: Including correlation analysis and principal component analysis (PCA).

## Key Findings

1. The most important features affecting AI detectability are:
   - Word overlap
   - POS tags (especially adjectives, punctuation, and adpositions)
   - Named entities (especially ordinal numbers and organizations)
   - Readability scores

2. Syntactic complexity and POS tags have the most significant impact on detectability, as revealed by our ablation study.

3. The average semantic similarity between original and undetectable texts is 0.8590, indicating substantial preservation of meaning.

4. Named entities and certain POS tags show the highest correlation with AI detectability score differences.

## Detailed Analysis

### Feature Importance

Top 10 most important features (Attention Model):
1. word_overlap: 0.0250
2. undetectable_pos_ADJ: 0.0184
3. original_pos_PUNCT: 0.0168
4. undetectable_ner_ORDINAL: 0.0158
5. undetectable_pos_CCONJ: 0.0157
6. original_pos_ADP: 0.0155
7. original_pos_ADV: 0.0155
8. original_flesch_reading_ease: 0.0149
9. undetectable_ner_ORG: 0.0143
10. original_pos_INTJ: 0.0135

### Ablation Study Results

1. Syntactic_complexity: Performance drop = 0.0114
2. POS_tags: Performance drop = 0.0029
3. Coherence: Performance drop = 0.0000
4. Sentiment: Performance drop = 0.0000
5. Readability: Performance drop = -0.0056
6. Paraphrasing: Performance drop = -0.0065
7. Named_entities: Performance drop = -0.0271

### BERT-based Semantic Analysis

- Average cosine similarity: 0.8590
- Minimum similarity: 0.1176
- Maximum similarity: 0.9947

### Correlation Analysis

Top correlations with AI detectability score difference:
1. undetectable_ner_ORG: 0.9964
2. undetectable_ner_PERCENT: 0.9964
3. original_ner_ORDINAL: 0.9963
4. undetectable_ner_PRODUCT: 0.9961
5. undetectable_ner_WORK_OF_ART: 0.9952

## Visualizations

![alt text](https://github.com/will4381/ai-detection-analysis/blob/main/images/top20features.png)

![alt text](https://github.com/will4381/ai-detection-analysis/blob/main/images/ablation.png)

![alt text](https://github.com/will4381/ai-detection-analysis/blob/main/images/cosine.png)

![alt text](https://github.com/will4381/ai-detection-analysis/blob/main/images/pca.png)

![alt text](https://github.com/will4381/ai-detection-analysis/blob/main/images/heatmap.png)

![alt text](https://github.com/will4381/ai-detection-analysis/blob/main/images/pca(2).png)

## Conclusions

Our comprehensive analysis provides deep insights into the features and patterns that contribute to making text appear more 'human-like' and less detectable as AI-generated. Key findings include:

1. The importance of word overlap, specific POS tags, and named entities in determining AI detectability.
2. The significant impact of syntactic complexity on text detectability.
3. High semantic similarity between original and 'undetectable' texts, suggesting effective preservation of meaning.
4. Complex feature interactions captured by our Feature Interaction Network.
5. Shared representations learned through multi-task learning, showing how different aspects of text generation and detection are related.

These results can inform strategies for creating more natural-sounding AI-generated text and improving AI detection methods. They also highlight the complexity of the task and the need for sophisticated, multi-faceted approaches to both text generation and detection.

## Future Work

1. Expand the dataset to include a wider variety of text types and sources.
2. Investigate the impact of context and document-level features on detectability.
3. Explore more advanced neural network architectures, such as transformers, for feature extraction and prediction.
4. Conduct user studies to correlate machine-based detectability scores with human perception of text naturalness.
5. Develop and test strategies for improving the 'human-likeness' of AI-generated text based on these findings.
