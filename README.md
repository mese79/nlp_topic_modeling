# NLP Topic Modeling
<br>

In this repository, I've run some topic modeling experiments on the *NIPS papers* dataset. 


<br><br>

### Requirements:
- python 3.x
- pandas
- nltk
- gensim
- pyLDAvis

<br><br>

### Usage
To prepare corpus and train LDA model, first unzip `data/papers.tar.xz` into the `data` folder and then:
```python
python topic_modeling_train.py
```
After training, you can see visualization results in `lda_vis.html` file.

To test model:
```python
python topic_modeling_test.py
```

There is also a jupyter notebook containing complete code for train and test process.  