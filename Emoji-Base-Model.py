#Emoji-Base-Model

import os
import argparse
import logging
import sklearn
import pandas as pd
import random
import emoji
import re
import torch.multiprocessing
from simpletransformers.t5 import T5Model, T5Args
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

def getPre(labels, preds):
    return precision_score(labels, preds, average="binary", pos_label='0')

def getRecall(labels, preds):
    return sklearn.metrics.recall_score(labels, preds, average=None)

def count_matches(labels, preds):
    #print("LABEL", labels)
    #print("PREDICTION", preds)
    return sum([1 if label == pred else 0 for label, pred in zip(labels, preds)])

def main():

    parser=argparse.ArgumentParser()
    parser.add_argument("gpu",type=int)
    #other argparse stuff for your program
    args=parser.parse_args()

    gpu=args.gpu
    assert gpu>=0 and gpu<4

    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    torch.multiprocessing.set_sharing_strategy('file_system')

    train_df = pd.read_csv('EMO_Training_BIG.csv', encoding='utf-8', sep=",")
    test_df = pd.read_csv('EMO_Testing_BIG.csv', sep=",")

    train_df = train_df.drop(train_df.columns[0], axis=1)
    test_df = test_df.drop(test_df.columns[0], axis=1)

    train_df = train_df.applymap(str)
    test_df = test_df.applymap(str)    
 
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)


    model_args = T5Args()
    model_args.num_train_epochs = 3
    model_args.no_save = False
    model_args.evaluate_generated_text = True
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_verbose = True
    model_args.overwrite_output_dir = True
    model_max_length=512
    model_args.accelerator="gpu"
    model_args.use_multiprocessing = False
    model_args.output_dir = "LARGE-Emoji-Base-Model"

    modelEmoReplace = T5Model("t5", "t5-base", args=model_args,use_cuda=False)
    modelEmoReplace.train_model(train_df, eval_data=test_df, matches=count_matches, acc=accuracy_score, rec=getRecall, pre=getPre)
    print("------------------THIS IS THE EMOJI BASE---------------")
    print(modelEmoReplace.eval_model(test_df, matches=count_matches, acc=accuracy_score, rec=getRecall, pre=getPre))

main()
