#!/usr/bin/env bash

mkdir embeddings
cd embeddings

# download glove embeddings
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip

#touch glove_vocab.txt
#while IFS=" " read col1 col2
#do
#   echo $col1 >> glove_vocab.txt
#done < glove.840B.300d.txt

# download counter-fitted embeddings
wget https://github.com/nmrksic/counter-fitting/raw/master/word_vectors/counter-fitted-vectors.txt.zip
unzip counter-fitted-vectors.txt.zip

cd ..

