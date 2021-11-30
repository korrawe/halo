import os, sys
import glob
import pandas as pd
import tqdm
import numpy as np

result_file = "/media/korrawe/data/works/gf/hp3d_fix/ho3d_correction/Batch_4293442_batch_results.csv"
# result_file = "/home/kkarunratanakul/works/userstudy/Batch_3944221_batch_results.csv"
# result_file = "/home/kkarunratanakul/works/userstudy/baseline_fhb_batch_results.csv"
# result_file = "/home/kkarunratanakul/works/userstudy/baseline_ho3d_batch_results.csv"
# result_file = "/home/kkarunratanakul/works/userstudy/baseline_obman_batch_results.csv"
# result_file = "/home/kkarunratanakul/works/userstudy/baseline_obman_batch_results.csv"


sampled_stat = []
obman_stat = []

userID = {}

with open(result_file, 'r') as f:
    header = f.readline()
    # print(header)
    i = 0
    for row in tqdm.tqdm(f):
        # print(row)
        cols = row.strip().replace('"', '').split(',')
        # print(cols)
        name = cols[-2]
        # print("name", name)
        score_txt = cols[-1]
        if score_txt == "strongly disagree": score = 1
        elif score_txt == "disagree": score = 2
        elif score_txt == "neither agree nor disagree": score = 3
        elif score_txt == "agree": score = 4
        elif score_txt == "strongly agree": score = 5

        # print("score", score)
        worker = cols[18]
        print("worker", worker)
        if not worker in userID:
            userID[worker] = 0
        else:
            # print("duppp")
            userID[worker] += 1

        image_name = name.split('/')[-1]
        # print("image name", image_name)
        # print(score)
        # if image_name[0] == '0':
        #     # print("obman", score)
        #     obman_stat.append(score)
        # else:
        #     sampled_stat.append(score)
        #     # print("ours", score)
        sampled_stat.append(score)
        
        # i += 1
        # if i > 50:
        #     break

print("unique user:", len(userID.keys()))
print("sampled mean:", np.mean(sampled_stat))
print("sampled sd:", np.std(sampled_stat))
print("number of sample hand:", len(sampled_stat) / 3.0)

print("----")
# print("ground truth obman mean:", np.mean(obman_stat))
# print("ground truth sd:", np.std(obman_stat))
# print("number of ground truth hand:", len(obman_stat) / 3.0)

print("total HIT", len(sampled_stat) + len(obman_stat))
print("total samples", (len(sampled_stat) + len(obman_stat)) / 3)
