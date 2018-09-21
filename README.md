# Contextualized Word Representations for Multi-Sense Embedding

## Summary
Generate multiple word representations for each word in dependency structure relations.

### Paper
Contextualized Word Representations for Multi-Sense Embedding

### Author/Author
- Kazuki Ashihara (Osaka University)
- Tomoyuki Kajiwara (Osaka University)
- Yuki Arase (Osaka University)
- Satoru Uchida (Kyushu University)


## Abstract
We propose methods to generate multiple word representations for each word based on the dependency structure relations.
In order to deal with the data sparseness problem due to the increase in the size of vocabulary, the initial value for each word representations is determined using the pre-trained word representations.
It is expected that the representations of low frequency words will remain in the vicinity of the initial value, which will in turn reduce the negative effects of data sparseness. 

## Novelty
- Capture word senses at a finer-grained level.
- Using dependency structure relations.
- Pre-training and Post-training.


## Proposed Method
<img width="237" alt="default" src="https://i.imgur.com/M8Vcpf1.png">
<img width="237" alt="default" src="file:///home/ashihara/PycharmProjects/picture/training.png">

## Result
- Context-Aware Word Similarity Task

- Lexical Substitution Task




## コメント




NL研に出すかも？しれない内容です。

# 概要
* pre-trainingを行い通常の分散表現を得る
* コーパス中から依存関係にある単語 (context-word) の抽出
* post-trainigでcontext-wordごとに語義を学習

# テストプレイ
* コーパス及びそれを形態素解析・依存構造解析したコーパスを用意。テストコーパスとして　`./corpus/`フォルダ内にデータを用意。
* `make_context_word.py`は`./corpus/corpus_parse.txt`からcontext-wordを抽出

1. `pre_train.py`を実行。`./model/pre_train.model`が生成される。
2. `make_context_word.py`を実行。
3. `post-training`を実行。`./model/post_train.model`が生成される。

`./model/post_train.model`には、context-wordごとに語義を割り当てた分散表現も含まれる。

今回は簡約化のためコードの一部は実際に使用したものとは異なる。


