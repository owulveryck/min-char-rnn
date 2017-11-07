This repository holds the basic implementation of a RNN.

It is based on the blog post [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) from Andrej Karpathy

The code is basically a transcript from his [gist](https://gist.github.com/karpathy/d4dee566867f8291f086).

I also got some help from Daniel Whitenack's [Building a Neural Net from Scratch in Go](http://www.datadan.io/building-a-neural-net-from-scratch-in-go/)

For more information, please refer to this [blog post](https://blog.owulveryck.info/2017/10/29/about-recurrent-neural-network-shakespeare-and-go.html)


# Configuration

## Hyper parameters of the neural nerwork 

```shell
RNN_INPUTNEURONS      Integer
RNN_OUTPUTNEURONS     Integer
RNN_HIDDENNEURONS     Integer    100        true
RNN_LEARNINGRATE      Float      1e-1       true
RNN_ADAGRADEPSILON    Float      1e-8       true
RNN_RANDOMFACTOR      Float      0.01
```

## Parameters of the executable

```shell
MIN_CHAR_SAMPLESIZE         Integer    100        true
MIN_CHAR_SAMPLEFREQUENCY    Integer    1000       true
MIN_CHAR_INFOFREQUENCY      Integer    100        true
MIN_CHAR_BACKUPFREQUENCY    Integer    1000       true
MIN_CHAR_BACKUPPREFIX       String
MIN_CHAR_BACKUPSUFFIX       String
```

## Parameters of the char codec

```shell
CHAR_CODEC_CHOICE     hard|soft (default hard)
CHAR_CODEC_EPOCH      100
CHAR_CODEC_VOCAB_FILE
CHAR_CODEC_INPUT_FILE
CHAR_CODEC_BATCHSIZE  default 25
```

# Usage

Example:

This will train the RNN with Shakespeare inputs and save every now and then the model to `shakespeare.bin`

```shell
export CHAR_CODEC_INPUT_FILE=data/shakespeare/input.txt
export CHAR_CODEC_VOCAB_FILE=data/shakespeare/vocab.txt
export RNN_ADAGRADEPSILON=1e-8
export RNN_RANDOMFACTOR=0.1
export RNN_LEARNINGRATE=1e-1
export MIN_CHAR_CHOICE=hard
export RNN_HIDDENNEURONS=66
export MIN_CHAR_BATCHSIZE=25
export MIN_CHAR_SAMPLEFREQUENCY=1000
export MIN_CHAR_EPOCHS=100
export MIN_CHAR_SAMPLESIZE=500
export MIN_CHAR_BACKUPPREFIX=shakespeare
export MIN_CHAR_BACKUPFREQUENCY=1000
export CHAR_CODEC_CHOICE=soft
echo "starting sequence for the sampling" | ./min-char-rnn -train
```

To use the pre-train model:

```
echo "Initial sample" | ./min-char-rnn -restore shakespeare.bin
```
