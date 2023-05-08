# EmoNet-for-EmotionDecoding

Note: This is vary early access of EmoNet! The code and instructions would be constantly updated.

EmoNet is a machine learing framework for spike decoding with confident learing component embedded. The basic idea is to enhance neural decoding robustness by cutting down uncertainty in samples. In our study, this framework outperformed ohter neural decoding SOTAs.

The confident learning component is implemented by cleanlab: https://github.com/cleanlab/cleanlab

Here is a simple example.

Example data: https://pan.baidu.com/s/1Xb31UaLrLy03k6Q-l0dwrw?pwd=8888

Download and unzip example datas, edit the dataset path in '\Dataset\DatasetTemplata.py',  run '\main.py' to see EmoNet's improvement in emotion mission.

You can embed your deep learning model into EmoNet framework following those steps:

step 1 : Set up your dataset and indeces.

step 2 : Establish your deep learning model and its epoch training function. 

step 3 : Pass your dataset, index, train function, model and other paremeters into TrainNet.train.train_EmoNet() and start your train.
