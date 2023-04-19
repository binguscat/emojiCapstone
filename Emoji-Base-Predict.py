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
    return sklearn.metrics.precision_score(labels, preds, average="binary", pos_label="0")

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

#read in the csv
    eval_df = pd.read_csv('data/emojiRemoveTest.csv', encoding='utf-8')

#strip the target text and create the to_predict dataFrame
    eval_df = eval_df.drop(eval_df.columns[0], axis=1)
    eval_df = eval_df.applymap(str)
    print(len(eval_df), " items in testing set\n")
    print("Remove Model Test 4")

#run model evaluation
    model2 = T5Model("t5", "./models/No-Emoji-Model", use_cuda=False) 
    print(model2.eval_model(eval_df, matches=count_matches, acc=accuracy_score, rec=getRecall, pre=getPre))
main()
