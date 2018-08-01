# SIGNL
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


