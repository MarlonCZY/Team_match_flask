# coding: utf-8

from __future__ import print_function

import pandas as pd
import numpy as np
import nltk
import string
import os
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import cPickle

import tfidf

def run():


    print("pickle start")
    # read and load data
    filename = '/Users/ziyuancui/Desktop/Job_Recommender_Deliver/brix/users_ai_with_cat.csv'
    file_path = "/Users/ziyuancui/Desktop/Job_Recommender_Deliver/brix/"
    df = pd.read_csv(filename,header=None,names=['id','name','title','description','location','rate','cat','skill'])

    # remove entries with NA in either Skills or Discription column
    df = df[~ (pd.isnull(df['skill']) | pd.isnull(df['description']))]
    #df.fillna('')

    # remove non ascii
    df['description'] = df['description'].apply(lambda x: re.sub(r'[^\x00-\x7F]+',' ', x))

    # remove empty description
    df = df[~df['description'].apply(lambda x: x.isdigit() | len(x)<10 )]



    df['country'] = df['location'].apply(lambda x:  re.split(',', re.split(u'\xa0',x)[0])[-1].strip() if isinstance(x,str) else '')

    # whether the user is native of US
    df['native'] = df['country'].apply(lambda x: True if x=='United States' else False)


    print("calculating")
    # tfidf, tfs on Skills
    tfidf_skills, tfs_skills = tfidf.get_tfidf(df['skill'].values, tfidf.tokenize_skill, max_features=2000)

    # tfidf, tfs on Skills
    tfidf_des, tfs_des = tfidf.get_tfidf(df['description'].values, tfidf.tokenize, max_features=5000)

    print("dumping")

    cPickle.dump(df[['id','cat','native']], open( file_path+"jobs_recomm_save_id.pkl", "wb" ))
    cPickle.dump(tfs_skills, open( file_path+"tfs.pkl", "wb" ))
    cPickle.dump(tfs_des, open( file_path+"tfs_des.pkl", "wb" ))
    cPickle.dump(tfidf_skills, open( file_path+"tfidf.pkl", "wb" ))
    cPickle.dump(tfidf_des, open( file_path+"tfidf_des.pkl", "wb" ))

    print("finish pickle")


if __name__=="__main__":
    run()



