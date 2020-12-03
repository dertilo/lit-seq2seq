# lit-seq2seq
pytorch lightning seq2seq

### idea
1. steal [tutorial-chatbot](https://github.com/pytorch/tutorials/blob/master/beginner_source/chatbot_tutorial.py)
2. refactor it + use pytorch-lightning
* use [pylama](https://github.com/klen/pylama)

### setup
* local
```shell script
conda env create -f environment.yml
wget http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip
unzip cornell_movie_dialogs_corpus.zip -d data
```
* google colab