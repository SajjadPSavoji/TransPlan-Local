from Libs import *
from Utils import *


import glob
from operator import index
import re
import time
import json
import ctypes
from collections import defaultdict
import cv2
import matplotlib
import numpy as np
# from pymatreader import read_mat
import pandas as pd
from matplotlib import cm
from counting.resample_gt_MOI.resample_typical_tracks import track_resample
from tqdm import tqdm
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, AgglomerativeClustering, AffinityPropagation
from sklearn.cluster import SpectralClustering
import pickle as pkl
import copy
import sys

# add your clustering algorithms here
# follow a sklearn-like API for consistency
# this means that your clustering object should have the fit_predict method

clusterers = {}
clusterers["DBSCAN"] = DBSCAN(eps = 0.8, min_samples=2, metric="precomputed")
clusterers["SpectralKNN"] = SpectralClustering(affinity="precomputed_nearest_neighbors", n_clusters=12)
clusterers["SpectralFull"] = SpectralClustering(affinity="precomputed", n_clusters=12)

def fit_circle(traj):
    x = traj[:, 0]
    y = traj[:, 1]
    x_m = np.mean(x)
    y_m = np.mean(y)
    u = x - x_m
    v = y - y_m
    Suv  = np.sum(u*v)
    Suu  = np.sum(u**2)
    Svv  = np.sum(v**2)
    Suuv = np.sum(u**2 * v)
    Suvv = np.sum(u * v**2)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)

    # Solving the linear system
    A = np.array([ [ Suu, Suv ], [Suv, Svv]])
    B = np.array([ Suuu + Suvv, Svvv + Suuv ])/2.0
    uc, vc = np.linalg.solve(A, B)

    xc_1 = x_m + uc
    yc_1 = y_m + vc

    # Calculation of all distances from the center (xc_1, yc_1)
    Ri_1      = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
    R_1       = np.mean(Ri_1)
    # residu_1  = np.sum((Ri_1-R_1)**2)
    # residu2_1 = np.sum((Ri_1**2-R_1**2)**2)
    return xc_1, yc_1, R_1 

# can be used for calculating descrete diff of a trajectory
def diff_traj(traj):
    V = []
    # skip the first point
    for i in range(1, len(traj)):
        # assuming traj[index] is np array
        p1, p2 = traj[i-1], traj[i]
        v = p2-p1
        V.append(v)
    return V

def accelate_vector(traj_a):
    v_a = diff_traj(traj_a)
    a_a = diff_traj(v_a)
    a_a = np.mean(a_a, axis=0)
    if np.linalg.norm(a_a)>0:
        a_a = a_a/np.linalg.norm(a_a)
    return a_a
    
def viz_CMM(current_track):
    image_path = "./../../Dataset/DundasStAtNinthLine.jpg"
    img = cv.imread(image_path)
    rows, cols, dim = img.shape
    for p in current_track:
        x, y = int(p[0]), int(p[1])
        img = cv.circle(img, (x,y), radius=2, color=(70, 255, 70), thickness=2)
    cv.imshow('image',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def viz_all_tracks(all_tracks, labels, save_path, image_path):
    img = cv.imread(image_path)
    rows, cols, dim = img.shape

    for current_track, current_label in zip(all_tracks, labels):
        if current_label<0: continue
        for p in current_track:
            x, y = int(p[0]), int(p[1])
            np.random.seed(int(current_label))
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            img = cv.circle(img, (x,y), radius=2, color=color, thickness=2)

    total_clusters = int(len(np.unique(labels)))
    # add description for the visualization on the top part of image
    text = f"{total_clusters} clusters annotated with different colors"
    coordinates = (15,15)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (0,0,0)
    thickness = 1
    img = cv2.putText(img, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
    
    cv.destroyAllWindows()
    if os.path.exists(save_path):
        os.remove(save_path)
    cv2.imwrite(save_path, img)


def group_tracks_by_id(df):
    # this function was writtern for grouping the tracks with the same id
    # usinig this one can load the data from a .txt file rather than .mat file
    all_ids = np.unique(df['id'].to_numpy(dtype=np.int64))
    data = {"id":[], "trajectory":[], "frames":[]}
    for idd in tqdm(all_ids):
        frames = df[df['id']==idd]["fn"].to_numpy(np.float32)
        id = idd
        trajectory = df[df['id']==idd][["x", "y"]].to_numpy(np.float32)
        
        data["id"].append(id)
        data["frames"].append(frames)
        data["trajectory"].append(trajectory)
    df2 = pd.DataFrame(data)
    return df2

# def cmm_distance(traj_a, traj_b, cmmlib):
#     traj_a = traj_a.astype(np.float64)
#     traj_b = traj_b.astype(np.float64)
#     if traj_a.shape[0] >= traj_b.shape[0]:
#         c = cmmlib.cmm_truncate_sides(traj_a[:, 0], traj_a[:, 1], traj_b[:, 0], traj_b[:, 1], traj_a.shape[0],
#                                         traj_b.shape[0])
#     else:
#         c = cmmlib.cmm_truncate_sides(traj_b[:, 0], traj_b[:, 1], traj_a[:, 0], traj_a[:, 1], traj_b.shape[0],
#                                         traj_a.shape[0])
#     return c


def find_centers_MNAVG(df, labels, M):
    unique_labels = np.unique(labels)
    center_indexes = []
    for ul in unique_labels:
        if ul < 0 : continue
        M_ul = M[labels==ul, :][:, labels==ul]
        indexes_ul = indexes[labels==ul].reshape(-1,)
        M_ul_avg = np.mean(M_ul, axis=0)
        i = np.argmin(M_ul_avg)
        center_indexes.append(int(indexes_ul[i]))
    return center_indexes

def find_centers_MXLEN(df, labels, M):
    unique_labels = np.unique(labels)
    center_indexes = []
    for ul in unique_labels:
        if ul < 0 : continue
        indexes_ul = indexes[labels==ul].reshape(-1,)
        traj_ul_len = [len(traj) for traj in df["trajectory"].iloc[indexes_ul]]
        i = np.argmax(traj_ul_len)
        center_indexes.append(int(indexes_ul[i]))
    return center_indexes

def cluster(args):
    # variables to be used from the args
        # args.ReprojectedPklMeter
        # args.ReprojectedPkl
        # args.ClusterMetric
    # save clustering result to the following args variables 
        # args.ReprojectedPklMeterCluster
        # args.ReprojectedPklCluster
        # args.ClusteringDistanceMatrix
        # args.ClusteringVis

    metric = Metric_Dict[args.ClusterMetric]
    df_meter_ungrouped = pd.read_pickle(args.ReprojectedPklMeter)
    df_reg_ungrouped   = pd.read_pickle(args.ReprojectedPkl)

    df_meter = group_tracks_by_id(df_meter_ungrouped)
    df_reg   = group_tracks_by_id(df_reg_ungrouped)
    
    df_meter_resampled = copy.deepcopy(df_meter)
    df_reg_resampled   = copy.deepcopy(df_reg)

    df_meter_resampled['trajectory'] = df_meter_resampled['trajectory'].apply(lambda x: track_resample(x).astype("float64"))
    df_reg_resampled['trajectory'] = df_reg_resampled['trajectory'].apply(lambda x: track_resample(x).astype("float64"))

    indexes = np.array([i for i in range(len(df_meter))]).reshape(-1, 1)

    M = None
    # check if the distance matrix is available first
    if False:
        pass
    # if os.path.exists(args.ClusteringDistanceMatrix):
    #     with open(args.ClusteringDistanceMatrix, "rb") as f:
    #         M = np.load(f)
    else:
        # compute distance matrix
        M = np.zeros(shape=(len(indexes), len(indexes)))
        for i in tqdm(indexes):
            for j in range(int(i)+1, len(indexes)):
                traj_a = df_meter_resampled['trajectory'].iloc[int(i)]
                traj_b = df_meter_resampled['trajectory'].iloc[int(j)]
                c =  metric(traj_a, traj_b)
                M[int(i), int(j)] = c
                M[int(j), int(i)] = c
        # with open(args.ClusteringDistanceMatrix, "wb") as f:
        #     np.save(f, M)

    clt = clusterers[args.ClusteringAlgo]
    if args.ClusteringAlgo == "SpectralFull":
        M = np.exp(-1*M)
    labels = clt.fit_predict(M)
    sns.histplot(labels)
    print(np.unique(labels))

    # save clustered trackes with labels for both meter and regular
    df_meter['cid'] = labels
    df_meter.to_pickle(args.ReprojectedPklMeterCluster)
    df_reg['cid'] = labels
    df_reg.to_pickle(args.ReprojectedPklCluster)

    viz_all_tracks(df_reg["trajectory"], labels, save_path = args.ClusteringVis, image_path=args.HomographyTopView)
    return SucLog("clustering executed successfully")

# center_indexes = find_centers_MNAVG(df, labels, M)
# viz_all_tracks(df["trajectory"].iloc[center_indexes], labels[center_indexes], save_path = "./../../Results/DBSCAN_centers_MAVD.png")
# center_indexes = find_centers_MXLEN(df, labels, M)
# viz_all_tracks(df["trajectory"].iloc[center_indexes], labels[center_indexes], save_path = "./../../Results/DBSCAN_centers_MXLN.png")

##### Agglomorative Clustering
# clt = AgglomerativeClustering(n_clusters = 12, affinity="precomputed", linkage='average')
# labels = clt.fit_predict(M)
# sns.histplot(labels)
# print(np.unique(labels))
# plt.show()
# print(labels)
# viz_all_tracks(df["trajectory"], labels, save_path = "./../../Results/Agglomorattive.png")

# center_indexes = find_centers_MNAVG(df, labels, M)
# viz_all_tracks(df["trajectory"].iloc[center_indexes], labels[center_indexes], save_path = "./../../Results/Agglomorative_centers_MAVD.png")
# center_indexes = find_centers_MXLEN(df, labels, M)
# viz_all_tracks(df["trajectory"].iloc[center_indexes], labels[center_indexes], save_path = "./../../Results/Agglomorative_centers_MXLN.png")

#### Affinity Propagation 
# clt = AffinityPropagation(damping =0.5 , affinity="precomputed", max_iter=2000)
# labels = clt.fit_predict(M)
# sns.histplot(labels)
# print(np.unique(labels))
# plt.show()
# print(labels)
# viz_all_tracks(df["trajectory"], labels, save_path = "./../../Results/AffinityPropagation.png")
# center_indexes = find_centers_MNAVG(df, labels, M)
# viz_all_tracks(df["trajectory"].iloc[center_indexes], labels[center_indexes], save_path = "./../../Results/Agglomorative_centers_MAVD.png")
# center_indexes = find_centers_MXLEN(df, labels, M)
# viz_all_tracks(df["trajectory"].iloc[center_indexes], labels[center_indexes], save_path = "./../../Results/Agglomorative_centers_MXLN.png")


# idea 1: for each cluster find the trajectory that has the least average distance from all the other clusters
# how to find cluster centers ??