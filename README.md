# What is the Multilingual-bist-parser?

It is a multi-source (many languages) trainable dependency parser extended by BIST-parser. 
If you want to know basic background of this parser, please refer to [Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations](https://www.transacl.org/ojs/index.php/tacl/article/viewFile/885/198). (We strongly recommend you to check the paper from Eliyahu Kiperwasser 2016). 

This parser was used for CoNLL 2017 shared task. we ranked at 5th among 33teams. For further information, please check [A System for Multilingual Dependency Parsing based on Bidirectional LSTM Feature Representations](http://www.aclweb.org/anthology/K17-3006)

This parser can train models both monolingual and multilingual way, but here we give you an example shell file (sme.sh) that is just for multilingual approach as a beginning step.
Note that!! Even if you train a model in a monolingual way, 
the output model would be different between original BIST-parser in terms of performance and usability because there are some differences.
 - Multilingualism: It takes language-hot encoding and multilingual word embedding as additional features.
 - Having a Unique ROOT: It has just one ROOT for addressing each sentence.  ref:[here](https://github.com/elikip/bist-parser/issues/10)
 - Training parsers with Universal Dependency (CoNLL-X format).
 - Several options for training different scenarios (use of XPOS and UPOS, concatenation of word embeddings or not, multilingual training ... so on)
 
 

## Dependencies

 * Python 2.7 interpreter
 * [DyNet 1.2 library](https://github.com/clab/dynet/tree/master/python)
  - boost-1.60.0, boost-1.58.0,  <B>boost-1.61.0 --> you can run it but need to fix some codes [here](https://github.com/elikip/bist-parser/issues/15)</B>
  - eigen hg clone https://bitbucket.org/eigen/eigen 
 
 Causion: based on version of Boost and Dynet, you may get different performance!!

## How to setup?

#### Dowonload the source code
```
    git clone git@github.com:jujbob/multilingual-bist-parser.git
```
#### setup Dynet
```
    sudo pip install cython 
    sudo apt-get install libboost-all-dev
    sudo pip install numpy
    cd dynet-base
    cd dynet
    mkdir build
    cd build
    sudo cmake .. -DEIGEN3_INCLUDE_DIR=../../eigen -DPYTHON=`which python`
    sudo make -j 2 
    cd python
    sudo python setup.py build
    sudo python setup.py install --user
```

On Macintosh we have succesfully installed Dynet and eigen following [these instructions](https://github.com/clab/dynet/blob/master/doc/source/python.rst#manual-installation) with Dynet version 2.0 and Boost version 1.6.

## How to train the parser?

#### An Example for training North Sami and Finnish
```
    vi train_bmst_multi_sme.sh   # change PRJ_DIR path for your home directory
     - PRJ_DIR=/home/ktlim/parser/multilingual-bist-parser
    vi training_file_list.txt    # change corpus path for your home directory
     - /home/ktlim/parser/multilingual-bist-parser/corpus/ud-treebanks-v2.0/UD_Finnish/fi-ud-train.conllu|||fi
     - /home/ktlim/parser/multilingual-bist-parser/corpus/ud-treebanks-v2.0/UD_North_Sami/sme-ud-train_200.conllu|||sme
     
    ./train_bmst_multi_sme.sh 
```


## Train your own models

You can train your own model by editing "train_bmst_multi_sme.sh" shell, 
and if you want to train a new combination of multilingual model, 
you may have multilingual word embedding and then also add related languages and codes to "language_vec.csv"

## Citation
we appreciate citing the following: 

Paper for 2017 CoNLL shared Task: Multilingual Parsing from Raw Text to Universal Dependencies
```
@article{lim2017system,
  title={A System for Multilingual Dependency Parsing based on Bidirectional LSTM Feature Representations},
  author={Lim, KyungTae and Poibeau, Thierry},
  journal={Proceedings of the CoNLL 2017 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies},
  volume={501},
  pages={63--70},
  year={2017}
}
```

## Contact

If you have any questions or suggestions, please send an email or write an issue

