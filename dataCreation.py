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

def getAccuracy(labels, preds):
    return sklearn.metrics.accuracy_score(labels, preds)

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

    Adv_df = pd.read_csv('russia.csv', encoding='utf-8')
    Con_df = pd.read_csv('tweets.tsv', sep="\t")

        

    Adv_jan = []
    Adv_feb = []
    Adv_mar = []
    Adv_apr = []
    Adv_may = []
    Adv_jun = []
    Adv_jul = []
    Adv_aug = []
    Adv_sep = []
    Adv_oct = []
    Adv_nov = []
    Adv_dec = []
    Adv_jan_2 = []
    Adv_feb_2 = []

    Con_jan = []
    Con_feb = []
    Con_mar = []
    Con_apr = []
    Con_may = []
    Con_jun = []
    Con_jul = []
    Con_aug = []
    Con_sep = []
    Con_oct = []
    Con_nov = []
    Con_dec = []
    Con_jan_2 = []
    Con_feb_2 = []

    
    count = 0
    for i in range(0, 1408711):
        if emoji.emoji_count(Adv_df["tweet_text"][i]) > 0:
            count+=1
    
    percent = round(((count / 1408711) * 100), 2)
    print("Count of Tweets with Emojis from Adversarial:", count)
    print("Percent of Tweets with Emojis from Adversarial:", percent,"%")
    count2 = 0
    for i in range(0, 551760):
        if emoji.emoji_count(Con_df["tweet_text"][i]) > 0:
            count2+=1  
    percent = round(((count2 / 551760) * 100), 2)
    print("Count of Tweets with Emojis from Control:", count2)
    print("Percent of Tweets Emojis from Control:", percent,"%")
    input()
    
    for i in range(0, 1408711):
        if '2016-01' in Adv_df["tweet_time"][i]:
            Adv_jan.append(Adv_df["tweet_text"][i])
        elif '2016-02' in Adv_df["tweet_time"][i]:
            Adv_feb.append(Adv_df["tweet_text"][i])    
        elif '2016-03' in Adv_df["tweet_time"][i]:
            Adv_mar.append(Adv_df["tweet_text"][i]) 
        elif '2016-04' in Adv_df["tweet_time"][i]:
            Adv_apr.append(Adv_df["tweet_text"][i]) 
        elif '2016-05' in Adv_df["tweet_time"][i]:
            Adv_may.append(Adv_df["tweet_text"][i])    
        elif '2016-06' in Adv_df["tweet_time"][i]:
            Adv_jun.append(Adv_df["tweet_text"][i]) 
        elif '2016-07' in Adv_df["tweet_time"][i]:
            Adv_jul.append(Adv_df["tweet_text"][i])
        elif '2016-08' in Adv_df["tweet_time"][i]:
            Adv_aug.append(Adv_df["tweet_text"][i]) 
        elif '2016-09' in Adv_df["tweet_time"][i]:
            Adv_sep.append(Adv_df["tweet_text"][i]) 
        elif '2016-10' in Adv_df["tweet_time"][i]:
            Adv_oct.append(Adv_df["tweet_text"][i])    
        elif '2016-11' in Adv_df["tweet_time"][i]:
            Adv_nov.append(Adv_df["tweet_text"][i]) 
        elif '2016-12' in Adv_df["tweet_time"][i]:
            Adv_dec.append(Adv_df["tweet_text"][i])
        elif '2017-01' in Adv_df["tweet_time"][i]:
            Adv_jan_2.append(Adv_df["tweet_text"][i]) 
        elif '2017-02' in Adv_df["tweet_time"][i]:
            Adv_feb_2.append(Adv_df["tweet_text"][i])

    for i in range(0, 551760):
        if '2016-01' in Con_df["date"][i]:
            Con_jan.append(Con_df["tweet_text"][i])
        elif '2016-02' in Con_df["date"][i]:
            Con_feb.append(Con_df["tweet_text"][i])    
        elif '2016-03' in Con_df["date"][i]:
            Con_mar.append(Con_df["tweet_text"][i]) 
        elif '2016-04' in Con_df["date"][i]:
            Con_apr.append(Con_df["tweet_text"][i]) 
        elif '2016-05' in Con_df["date"][i]:
            Con_may.append(Con_df["tweet_text"][i])    
        elif '2016-06' in Con_df["date"][i]:
            Con_jun.append(Con_df["tweet_text"][i]) 
        elif '2016-07' in Con_df["date"][i]:
            Con_jul.append(Con_df["tweet_text"][i])
        elif '2016-08' in Con_df["date"][i]:
            Con_aug.append(Con_df["tweet_text"][i]) 
        elif '2016-09' in Con_df["date"][i]:
            Con_sep.append(Con_df["tweet_text"][i]) 
        elif '2016-10' in Con_df["date"][i]:
            Con_oct.append(Con_df["tweet_text"][i])    
        elif '2016-11' in Con_df["date"][i]:
            Con_nov.append(Con_df["tweet_text"][i]) 
        elif '2016-12' in Con_df["date"][i]:
            Con_dec.append(Con_df["tweet_text"][i])
        elif '2017-01' in Con_df["date"][i]:
            Con_jan_2.append(Con_df["tweet_text"][i]) 
        elif '2017-02' in Con_df["date"][i]:
            Con_feb_2.append(Con_df["tweet_text"][i])

    sampleSize = 500

    Adv_jan = random.sample(Adv_jan, sampleSize)
    Adv_feb = random.sample(Adv_feb, sampleSize)
    Adv_mar = random.sample(Adv_mar, sampleSize)
    Adv_apr = random.sample(Adv_apr, sampleSize)
    Adv_may = random.sample(Adv_may, sampleSize)
    Adv_jun = random.sample(Adv_jun, sampleSize)
    Adv_jul = random.sample(Adv_jul, sampleSize)
    Adv_aug = random.sample(Adv_aug, sampleSize)
    Adv_sep = random.sample(Adv_sep, sampleSize)
    Adv_oct = random.sample(Adv_oct, sampleSize)
    Adv_nov = random.sample(Adv_nov, sampleSize)
    Adv_dec = random.sample(Adv_dec, sampleSize)
    Adv_jan_2 = random.sample(Adv_jan_2, sampleSize)
    Adv_feb_2 = random.sample(Adv_feb_2, sampleSize)

    Con_jan = random.sample(Con_jan, sampleSize)
    Con_feb = random.sample(Con_feb, sampleSize)
    Con_mar = random.sample(Con_mar, sampleSize)
    Con_apr = random.sample(Con_apr, sampleSize)
    Con_may = random.sample(Con_may, sampleSize)
    Con_jun = random.sample(Con_jun, sampleSize)
    Con_jul = random.sample(Con_jul, sampleSize)
    Con_aug = random.sample(Con_aug, sampleSize)
    Con_sep = random.sample(Con_sep, sampleSize)
    Con_oct = random.sample(Con_oct, sampleSize)
    Con_nov = random.sample(Con_nov, sampleSize)
    Con_dec = random.sample(Con_dec, sampleSize)
    Con_jan_2 = random.sample(Con_jan_2, sampleSize)
    Con_feb_2 = random.sample(Con_feb_2, sampleSize)

    Adv_jan = pd.DataFrame(Adv_jan)
    Adv_feb = pd.DataFrame(Adv_feb)
    Adv_mar = pd.DataFrame(Adv_mar)
    Adv_apr = pd.DataFrame(Adv_apr)
    Adv_may = pd.DataFrame(Adv_may)
    Adv_jun = pd.DataFrame(Adv_jun)
    Adv_jul = pd.DataFrame(Adv_jul)
    Adv_aug = pd.DataFrame(Adv_aug)
    Adv_sep = pd.DataFrame(Adv_sep)
    Adv_oct = pd.DataFrame(Adv_oct)
    Adv_nov = pd.DataFrame(Adv_nov)
    Adv_dec = pd.DataFrame(Adv_dec)
    Adv_jan_2 = pd.DataFrame(Adv_jan_2)
    Adv_feb_2 = pd.DataFrame(Adv_feb_2)

    Con_jan = pd.DataFrame(Con_jan)
    Con_feb = pd.DataFrame(Con_feb)
    Con_mar = pd.DataFrame(Con_mar)
    Con_apr = pd.DataFrame(Con_apr)
    Con_may = pd.DataFrame(Con_may)
    Con_jun = pd.DataFrame(Con_jun)
    Con_jul = pd.DataFrame(Con_jul)
    Con_aug = pd.DataFrame(Con_aug)
    Con_sep = pd.DataFrame(Con_sep)
    Con_oct = pd.DataFrame(Con_oct)
    Con_nov = pd.DataFrame(Con_nov)
    Con_dec = pd.DataFrame(Con_dec)
    Con_jan_2 = pd.DataFrame(Con_jan_2)
    Con_feb_2 = pd.DataFrame(Con_feb_2)

    trueSet = [Adv_jan, Adv_feb, Adv_mar, Adv_apr, Adv_may, Adv_jun, Adv_jul, Adv_aug, Adv_sep, Adv_oct, Adv_nov, Adv_dec, Adv_jan_2, Adv_feb_2]
    falseSet = [Con_jan, Con_feb, Con_mar, Con_apr, Con_may, Con_jun, Con_jul, Con_aug, Con_sep, Con_oct, Con_nov, Con_dec, Con_jan_2, Con_feb_2]

    trueDF = pd.concat(trueSet, ignore_index=True, sort=False)
    falseDF = pd.concat(falseSet, ignore_index=True, sort=False)

    #trueDF.columns = ['prefix', 'input_text', 'target_text']
    column_head = ["input_text"]
    trueDF.columns = column_head
    falseDF.columns = column_head
    trueDF["target_text"]  = '1'
    trueDF["prefix"] = "binary classification"
    falseDF["target_text"]  = '0'
    falseDF["prefix"] = "binary classification"

    
    fullSet = [trueDF, falseDF]
    fullDF = pd.concat(fullSet, ignore_index=True, sort=False)
    fullDF = fullDF.sample(frac = 1)

    full_train_DF = fullDF.sample(frac=0.7, random_state=28)
    full_test_DF = fullDF.drop(full_train_DF.index)


#strip http from data

    for index, row in fullDF.iterrows():
        row["input_text"] = re.sub(r'http\S+', '', row["input_text"])
        

#create the dataFrames
    emojiRemoveTest = full_test_DF.copy(deep=True)
    emojiReplaceTest = full_test_DF.copy(deep=True)
    emojiControlTest = full_test_DF.copy(deep=True)

#remove emojis from the dataset
    for index, row in emojiRemoveTest.iterrows():
        row["input_text"] = emoji.replace_emoji(row["input_text"], replace='')

#replace emojis in the dataset
    for index, row in emojiReplaceTest.iterrows():
        row["input_text"] = emoji.demojize(row["input_text"])

# cast dataframes to files
    emojiRemoveTest.to_csv("emojiRemoveTest3.csv")    
    emojiReplaceTest.to_csv("emojiReplaceTest3.csv")
    emojiControlTest.to_csv("emojiControlTest3.csv")


main()

