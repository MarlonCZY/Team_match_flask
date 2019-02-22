from flask import Flask
import json
from flask import Flask
from flask import request, Response
import generate_tfidf_models
import pandas as pd
import numpy as np
import csv
import nltk
import string
import os
import re
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import sys
import json
import pickle
import cPickle
import tfidf

app = Flask(__name__)

users = pickle.load(open('/Users/ziyuancui/Desktop/Job_Recommender_Deliver/brix/jobs_recomm_save_id.pkl', "rb"))
tfs_skill = pickle.load(open('/Users/ziyuancui/Desktop/Job_Recommender_Deliver/brix/jobs_recomm_save_tfs_skill.pkl', "rb"))
tfidf_skill = pickle.load(open('/Users/ziyuancui/Desktop/Job_Recommender_Deliver/brix/jobs_recomm_save_tfidf_skill.pkl', "rb"))
tfs_des = pickle.load(open('/Users/ziyuancui/Desktop/Job_Recommender_Deliver/brix/jobs_recomm_save_tfs_des.pkl', "rb"))
tfidf_des = pickle.load(open('/Users/ziyuancui/Desktop/Job_Recommender_Deliver/brix/jobs_recomm_save_tfidf_des.pkl', "rb"))

# reset index of users dataframe to default
users.reset_index(inplace=True, drop=True)


@app.route('/test')
def hello():
    return 'test hello world'


@app.route('/test1')
def testAdd():
    return "TEST ADDED"


"""
Request: 
{"name":"Marlon", "title":"web developer", 
"description":"I am professional.", "location":"Irvine",
"salary":20, "category":"Web Expert",
 "skills":"full stack    Javascript    software development"}

 Function: add user information into users_ai_with_cat.csv"

 Response: {"type":"Success"}
"""


@app.route('/addUser', methods=['POST'])
def add_user():
    data = request.data
    json_data = json.loads(data)
    #   use json format
    #   data = request.get_json()

    rowCount = 0

    file_path = "/home/ubuntu/brix/"
    # count the number of users
    with open(file_path + "users_ai_with_cat.csv", "r") as file:
        rowCount = len(file.readlines())
        print rowCount
    with open(file_path + "users_ai_with_cat.csv", "a") as file:
        writer = csv.writer(file)
        # [id, name, title, description, location, salary, category, skills]
        writer.writerow(
            [rowCount + 2, json_data['name'], json_data['title'], json_data['description'], json_data['location'],
             json_data['salary'], json_data['category'],
             json_data['skills']]
        )
    generate_tfidf_models.run()

    result = {"type": "success"}

    return Response(json.dumps(result), mimetype='application/json')


@app.route('/', methods=['GET', 'POST'])
def match():
    if request.method == 'POST':

        data = request.data
        json_data = json.loads(data)

        project_skills = json_data['skill']
        project_description = json_data['des']
        cat = json_data['category']
        location = json_data['location']

        # identify users with specific category
        if (cat.lower() == 'all') and (location.lower() == 'all'):
            uid = users['id'].values
            used_tfs_des = tfs_des
            used_tfs_skill = tfs_skill
        else:
            if cat.lower() == 'all':
                cond = pd.Series([True] * len(users))
            else:
                cond = users['cat'].apply(lambda x: x.lower() == cat.lower())

            if location.lower() == 'native':
                cond = cond & users['native']
            elif location.lower() == 'international':
                cond = cond & (~users['native'])

            index = users.index[cond]
            if len(index) < 1:
                index = users.index

            # get new set of users
            uid = users.iloc[index]['id'].values
            print(tfs_des)
            used_tfs_des = tfs_des[index]
            used_tfs_skill = tfs_skill[index]

        # individual match
        results, sc = tfidf.get_top_k_matches('  '.join(project_skills.split(',')), tfidf_skill, used_tfs_skill)
        results_des, sc_des = tfidf.get_top_k_matches(project_description, tfidf_des, used_tfs_des)

        # team match
        team_match = tfidf.get_top_k_nonoverlapping_matches('  '.join(project_skills.split(',')), tfidf_skill,
                                                            used_tfs_skill, K=5, non_negative=True)

        total_sc = sc * 0.75 + sc_des * 0.25
        total_rank = np.argsort(-total_sc.flatten())[:5]

        out = {}
        out['individual index'] = ','.join([str(uid[i]) for i in total_rank])
        out['individual score'] = ','.join(['{:.4f}'.format(total_sc[i]) for i in total_rank])

        out['team index'] = ','.join([str(uid[i]) for i, s in team_match])
        out['team score'] = ','.join(['{:.4f}'.format(s) for i, s in team_match])

        return Response(json.dumps(out), mimetype='application/json')


if __name__ == '__main__':
    app.run(debug=True)
