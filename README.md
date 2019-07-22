# Text Classification - Predicting occupations (EDAN70 - Project)

This is a neural network algorithm for predicting occupation based on text descriptions on persons.

## Getting Started

Before running any files you need to download the docria wikipedia file from:
http://fileadmin.cs.lth.se/nlp/wiki2018/enwiki.tar
Put this file in corpus/enwiki.

You also need to download the word embeddings from gloVe. Put the file name glove.6B.100d.txt in the root folder.
Download it here: https://nlp.stanford.edu/projects/glove/

Also delete all temp files named temp in all folders.

### Prerequisites

Install these libraries before moving on.

```
pip3 install keras
pip3 install pickle
pip3 install sklearn
```

### FETCHING DATA FROM WIKIDATA

The first thing you need to do is extract data from wikidata and match it with the docria file you downloaded.
To do this run first:

```
python extract_wikidata.py
```

And then

```
python match.py
```

### Preprocessing

Now you are ready to preprocess the data. Do this by running:

```
python preprocess.py
```

You will now be prompted if you want to create a big or small model.

### Building the model

Build the model by running:

```
python build_model.py
```

Once again you will be prompted which set you want to use.

### Evaluating and ploting confusion matrix

```
python evaluate_model.py
```
