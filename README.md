### How Important Is POS to Dependency Parsing? Joint POS Tagging and Dependency Parsing Neural Networks [[pdf](https://link.springer.com/chapter/10.1007%2F978-3-030-32381-3_50)]

# Introduction

It is widely accepted that part-of-speech (POS) tagging and dependency parsing are highly related. Most state-of-the-art dependency parsing methods still rely on the results of POS tagging, though the tagger is not perfect yet.

Moreover, it still remains uncertain about how important POS tagging is to dependency parsing. In this work, we propose a method to jointly learn POS tagging and dependency parsing so as to alleviate the error propagation problems.

# Inspiration

Currently, POS taggers are not perfect yet, and the taggers may be irrelevant to the sentences to be parsed. Therefore, we attempt to import POS information to improve dependency parsing. Our work supports our idea of joint learning.

---

# Method

The overall framework is shown in the figure. In this figure, the model is divided into three parts. First, we combine three different lexical information (character, word, and N-gram) to generate every word vector representations. Then, because our dependency parsing model is based on POS tagging results, in this stage we train POS tagging model as an intermediate helper, so as to dig in more lingustic information from the texts. Last, the input of dependency parsing is the concat of POS tagging results and sentence representations used in POS tagging.

![framework](https://github.com/hsuehkuan-lu/JointParser/blob/master/framework.png)

Our method uses Bidirectional Long Short-Term Memory neural networks on both POS tagging and dependency parsing tasks. And we use transition-based dependency parsing algorithm to generate parses.

---

# Experiment

In this work, we use [Universal Dependencies](https://universaldependencies.org/)(1.2 and 2.0) as our evaluating dataset.

In the experiment of UD 1.2, we compare our method with other three joint learning models, and our method outperforms all other methods. The results are reported in the following table, the scores are evaluated in LAS metric:

<img src="https://github.com/hsuehkuan-lu/JointParser/blob/master/experiment.png" width="800">

---

# Released Model

Our method is implemented in python3 with tensorflow.

Entry: `parser_model_las.py`

## Recommend Environment

Python 3.5 + tensorflow 1.12.0

## Model

[Download](https://drive.google.com/drive/folders/1kfX0DMxtegLTzrVR2qNWvxrf5_fpMydW?usp=sharing)
In the folder `model`, there are two trained models including English and Chinese dependency parsing models.

## Functions

1. Training model.
2. Evaluate dependency parsing (with CoNLL-U format).
3. Inference dependency parsing.

### Training

Training dependency parsing model. `train` function.

**Parameters**
1. --gpu: Assign gpu id.
2. --model_path: Assign model path.
3. --emb_file: Pre-trained embeddings path (optional).
4. --debug: Debug mode.
5. --train_file: Training set data path.
6. --dev_file: Development set data path.
7. --test_file: Testing set data path.

* Input data format should be [CoNLL-U](https://universaldependencies.org/format.html) format.

**Example**
```
python parser_model_las.py train 
  --gpu 0 \
  --model_path /data/DependencyParsing/model \
  --emb_file /data/glove.5B.50d.txt \
  --train_file /data/DependencyParsing/train.conll \
  --dev_file /data/DependencyParsing/dev.conll \
  --test_file /data/DependencyParsing/test.conll
```

---

### Evaluating

Evaluating dependency parsing data. `evaluate` function.

**Parameters**
1. --gpu: Assign gpu id.
2. --model_path: Assign model path.
3. --output_path: Ouput results data path.
4. --debug: Debug mode.
5. --test_file: Testing set data path.

**Example**
```
python parser_model_las.py evaluate 
  --gpu 0 \
  --model_path /data/DependencyParsing/model \
  --output_path /data/DependencyParsing/result \
  --test_file /data/DependencyParsing/test.conll
```

---

### Inference

Inference dependency parsing with given text or sentence. `inference` function.

**Parameter**
1. --gpu: Assign gpu id.
2. --model_path: Assing model path.

**Example**
```
python parser_model_las.py inference 
  --gpu 0 \
  --model_path /data/DependencyParsing/model 
```

Detailed model loading can reference function `inference` in `parser_model_las.py`. Open API is `parse_sents`.

`parse_sents`:
- Input: 
  - List of words:
  `[["Today", "is", "nice", "weather"]]`.
  
- Output:
  - POS tagging:
  `>>> [['t', 'v', 'n', 'n']]`
  - Dependency parsing:
  `>>> [[('weather', 'Today', 'nsubj'), ('weather', 'is', 'cop'), ('weather', 'nice', 'amod'), ('ROOT', 'weather', 'root')]]`

---

# Citation

If you found this work helpful, consider citing the work as:

```
@InProceedings{
  10.1007/978-3-030-32381-3_50,
  author="Lu, Hsuehkuan
  and Hou, Lei
  and Li, Juanzi",
  editor="Sun, Maosong
  and Huang, Xuanjing
  and Ji, Heng
  and Liu, Zhiyuan
  and Liu, Yang",
  title="How Important Is POS to Dependency Parsing? Joint POS Tagging and Dependency Parsing Neural Networks",
  booktitle="Chinese Computational Linguistics",
  year="2019",
  publisher="Springer International Publishing",
  address="Cham",
  pages="625--637",
  isbn="978-3-030-32381-3"
}
```

---