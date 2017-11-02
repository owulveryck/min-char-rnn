This repository holds the basic implementation of a RNN.

It is based on the blog post [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) from Andrej Karpathy

The code is basically a transcript from his [gist](https://gist.github.com/karpathy/d4dee566867f8291f086).

I also got some help from Daniel Whitenack's [Building a Neural Net from Scratch in Go](http://www.datadan.io/building-a-neural-net-from-scratch-in-go/)


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

