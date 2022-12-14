tracks_path = "./../Dataset/DandasStAtNinthLineFull/Results/Tracking/video.tracking.detectron2.ByteTrack.reprojected.pkl"
tracks_meter_path = "./../Dataset/DandasStAtNinthLineFull/Results/Tracking/video.tracking.detectron2.ByteTrack.reprojected.meter.pkl"
top_image = "./../Dataset/DandasStAtNinthLineFull/video.homography.top.png"
meta_data = "./../Dataset/DandasStAtNinthLineFull/video.metadata.json"
HomographyNPY = "./../Dataset/DandasStAtNinthLineFull/Results/Homography/video.homography.npy"



# tracks_path = "./../Dataset/SOW_src2/Results/Tracking/src2.tracking.detectron2.ByteTrack.reprojected.pkl"
# tracks_meter_path = "./../Dataset/SOW_src2/Results/Tracking/src2.tracking.detectron2.ByteTrack.reprojected.meter.pkl"
# top_image = "./../Dataset/SOW_src2/src2.homography.top.png"
# meta_data = "./../Dataset/SOW_src2/src2.metadata.json"
# HomographyNPY = "./../Dataset/SOW_src2/Results/Homography/src2.homography.npy"


from counting.resample_gt_MOI.resample_typical_tracks import track_resample

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy import optimize
import json
import sympy

from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, KMeans
import cv2 as cv

from sklearn.neighbors import KernelDensity as KDE
import sklearn

class MyPoly():
    def __init__(self, roi):
        self.poly = sympy.Polygon(*roi)
        self.lines = []
        for i in range(len(roi)-1):
            self.lines.append(sympy.Line(sympy.Point(roi[i]), sympy.Point(roi[i+1])))
        self.lines.append(sympy.Line(sympy.Point(*roi[-1]), sympy.Point(*roi[0])))
        
    def distance(self, point):
        p = sympy.Point(*point)
        distances = []
        for line in self.lines:
            distances.append(float(line.distance(p)))
        distances = np.array(distances)
        min_pos = np.argmin(distances)
        return distances[min_pos], int(min_pos)

    @property
    def area(self):
        return float(self.poly.area)

def is_monotonic(traj):
    orgin = traj[0]
    max_distance = -1
    for p in traj:
        dp = np.linalg.norm(p - orgin)
        if dp < max_distance: return False
        max_distance = dp
    return True

class IKDE():
    def __init__(self, kernel="exponential", bandwidth=1):
        self.kernel = kernel
        self.bw = bandwidth

    def fit(self, tracks):
        self.mois = np.unique(tracks["moi"])
        self.kdes = {}
        for moi in self.mois:
            self.kdes[moi] = sklearn.neighbors.KernelDensity(kernel=self.kernel, bandwidth=self.bw)
        for moi in self.mois:
            kde_data = []
            for i, row in tracks[tracks["moi"] == moi].iterrows():
                traj_len = len(row["trajectory"])
                for x, y in row["trajectory"]:
                    kde_data.append([x, y, i/traj_len])
            kde_data = np.array(kde_data)
            self.kdes[moi].fit(kde_data)

    def get_traj_score(self, traj, moi):
        traj_data = []
        for i in range(len(traj)):
            x, y = traj[i]
            traj_data.append([x, y, i/len(traj)])
        traj_data = np.array(traj_data)
        return np.sum(self.kdes[moi].score_samples(traj_data))
    
    def predict_traj(self, traj):
        moi_scores = []
        for moi in self.mois:
            moi_scores.append(self.get_traj_score(traj, moi))
        max_moi = self.mois[np.argmax(moi_scores)]
        return max_moi

    def predict_tracks(self, tracks):
        max_mois = []
        for i, row in tracks.iterrows():
            traj = row["trajectory"]
            max_mois.append(self.predict_traj(traj))
        return max_mois


def fit_circle(traj):
    def calc_R(xc, yc):
        """ calculate the distance of each data points from the center (xc, yc) """
        return np.sqrt((x-xc)**2 + (y-yc)**2)
    
    def f_2b(c):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()
    
    def Df_2b(c):
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc     = c
        df2b_dc    = np.empty((len(c), x.size))
    
        Ri = calc_R(xc, yc)
        df2b_dc[0] = (xc - x)/Ri                   # dR/dxc
        df2b_dc[1] = (yc - y)/Ri                   # dR/dyc
        df2b_dc    = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]
    
        return df2b_dc
    x = traj[:, 0]
    y = traj[:, 1]
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center_2b, ier = optimize.leastsq(f_2b, center_estimate, Dfun=Df_2b, col_deriv=True)
    
    xc_2b, yc_2b = center_2b
    Ri_2b        = calc_R(*center_2b)
    R_2b         = Ri_2b.mean()
    residu_2b    = np.sum((Ri_2b - R_2b)**2)
    return xc_2b, yc_2b, R_2b

# def fit_circle(traj):
#     x = traj[:, 0]
#     y = traj[:, 1]
#     x_m = np.mean(x)
#     y_m = np.mean(y)
#     u = x - x_m
#     v = y - y_m
#     Suv  = np.sum(u*v)
#     Suu  = np.sum(u**2)
#     Svv  = np.sum(v**2)
#     Suuv = np.sum(u**2 * v)
#     Suvv = np.sum(u * v**2)
#     Suuu = np.sum(u**3)
#     Svvv = np.sum(v**3)

#     # Solving the linear system
#     A = np.array([ [ Suu, Suv ], [Suv, Svv]])
#     B = np.array([ Suuu + Suvv, Svvv + Suuv ])/2.0
    
#     uc, vc = np.linalg.solve(A, B)

#     xc_1 = x_m + uc
#     yc_1 = y_m + vc

#     # Calculation of all distances from the center (xc_1, yc_1)
#     Ri_1      = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
#     R_1       = np.mean(Ri_1)
#     # residu_1  = np.sum((Ri_1-R_1)**2)
#     # residu2_1 = np.sum((Ri_1**2-R_1**2)**2)
#     return xc_1, yc_1, R_1 

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

def arc_length(track):
    """
    :param track: input track numpy array (M, 2)
    :return: the estimated arc length of the track
    """
    assert track.shape[1] == 2
    accum_dist = 0
    for i in range(1, track.shape[0]):
        dist_ = np.sqrt(np.sum((track[i] - track[i - 1]) ** 2))
        accum_dist += dist_
    return accum_dist



str_end_moi = {}
str_end_moi[(3, 0)] = 1
str_end_moi[(3, 1)] = 2
str_end_moi[(3, 2)] = 3
str_end_moi[(2, 3)] = 4
str_end_moi[(2, 0)] = 5
str_end_moi[(2, 1)] = 6
str_end_moi[(1, 2)] = 7
str_end_moi[(1, 3)] = 8
str_end_moi[(1, 0)] = 9
str_end_moi[(0, 1)] = 10
str_end_moi[(0, 2)] = 11
str_end_moi[(0, 3)] = 12

def str_end_to_moi(str, end):
    if (str ,end) in str_end_moi:
        return str_end_moi[(str, end)]
    return -1

    

moi_clusters = {}
moi_clusters[1] = 1
moi_clusters[2] = 2
moi_clusters[3] = 1
moi_clusters[4] = 1
moi_clusters[5] = 3
moi_clusters[6] = 1
moi_clusters[7] = 1
moi_clusters[8] = 2
moi_clusters[9] = 1
moi_clusters[10] = 1
moi_clusters[11] = 3
moi_clusters[12] = 1



with open(meta_data, 'r') as f:
    meta = json.load(f)
roi = meta["roi"]

M = np.load(HomographyNPY, allow_pickle=True)[0]
roi_rep = []
for p in roi:
    point = np.array([p[0], p[1], 1])
    new_point = M.dot(point)
    new_point /= new_point[2]
    roi_rep.append([new_point[0], new_point[1]])

# pg = sympy.Polygon(*roi)
pg = MyPoly(roi_rep)
print(pg.area)
th = 0.05 * np.sqrt(pg.area)
print(th)

tracks = group_tracks_by_id(pd.read_pickle(tracks_path))
tracks_meter = group_tracks_by_id(pd.read_pickle(tracks_meter_path))
tracks['index_mask'] = tracks_meter['trajectory'].apply(lambda x: track_resample(x, return_mask=True))

img = plt.imread(top_image)

img1 = cv.imread(top_image)
H, W, C = img1.shape
X_cord, Y_cord = np.meshgrid(range(int(W)), range(int(H)))
# X_cord, Y_cord = np.meshgrid(range(200, 600), range(200, 600))
Data_cord = np.stack([X_cord, Y_cord]).reshape(2, -1).T


# counter = 0
# mask = []
# i_strs = []
# i_ends = []
# moi = []

# for i, row in tqdm(tracks.iterrows(), total=len(tracks)):
#     traj = row["trajectory"]
#     d_str, i_str = pg.distance(traj[0])
#     d_end, i_end = pg.distance(traj[-1])
#     i_strs.append(i_str)
#     i_ends.append(i_end)
#     moi.append(str_end_to_moi(i_str, i_end))

#     if (d_str < th) and (d_end < th) and (not i_str == i_end) and is_monotonic(traj[row["index_mask"]]):
#         mask.append(True)
#         counter += 1
#         c=0
#         for x, y in traj:
#             x, y = int(x), int(y)
#             img1 = cv.circle(img1, (x,y), radius=1, color=(int(c/len(traj)*255), 70, int(255 - c/len(traj)*255)), thickness=1)
#             c+=1
#     else:
#         mask.append(False)

# print(counter/len(tracks))

# plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
# plt.show()

# tracks['i_str'] = i_strs
# tracks['i_end'] = i_ends
# tracks['moi'] = moi
# tracks_meter['i_str'] = i_strs
# tracks_meter['i_end'] = i_ends 
# tracks_meter['moi'] = moi


# ikde = IKDE()
# ikde.fit(tracks[mask])
# labels = ikde.predict_tracks(tracks)

# for i, row in tracks.iterrows():
#     traj = row["trajectory"]
#     img1 = cv.imread(top_image)
#     c=0
#     for x, y in traj:
#         x, y = int(x), int(y)
#         img1 = cv.circle(img1, (x,y), radius=1, color=(int(c/len(traj)*255), 70, int(255 - c/len(traj)*255)), thickness=2)
#         c+=1
#     plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
#     plt.title(f"label : {labels[i]}")
#     plt.show()

# exit()


# for k  in range(4):
#     for j in range(4):
#         if k == j: continue
#         img1 = cv.imread(top_image)
#         kde_data = []
#         for i, row in tqdm(tracks[mask].iterrows(), total=len(tracks[mask])):
#             traj = row["trajectory"]
#             # if is_monotonic(traj[row["index_mask"]]): 
#             #     continue
#             i_str = row["i_str"]
#             i_end = row["i_end"]
#             if i_str == k and i_end == j:
#                 c=0
#                 for x, y in traj:
#                     x, y = int(x), int(y)
#                     kde_data.append([x, y])
#                     img1 = cv.circle(img1, (x,y), radius=1, color=(int(c/len(traj)*255), 70, int(255 - c/len(traj)*255)), thickness=2)
#                     c+=1
#         # estimate probability distributions
#         kde_data = np.array(kde_data)
#         kde = KDE(kernel="exponential", bandwidth=1)
#         kde.fit(kde_data)
#         scores_cord = kde.score_samples(Data_cord).reshape(X_cord.shape)
#         # scores_cord = np.sum(Data_cord, axis=-1).reshape(X_cord.shape)
#         print(X_cord.shape, Y_cord.shape, Data_cord.shape, scores_cord.shape)
#         plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
#         plt.contourf(X_cord, Y_cord, scores_cord, alpha = 0.4, cmap='jet', levels=100)
#         plt.show()



train , test = sklearn.model_selection.train_test_split(tracks, test_size=0.2)
train_kde_data = []
test_kde_data = []
for i, row in train.iterrows():
    for p in row["trajectory"][row["index_mask"]]:
        train_kde_data.append(p)
for i, row in test.iterrows():
    for p in row["trajectory"][row["index_mask"]]:
        test_kde_data.append(p)


kernels  =["gaussian"]
costs_train = {}
costs_test  = {}
for ker in kernels:
    costs_train[ker] = []
    costs_test[ker]  = []
bws = np.linspace(2.5, 4, 50)

for ker in kernels:
    for bw in bws:
        kde = KDE(kernel=ker, bandwidth=bw)
        kde.fit(train_kde_data)
        score_train = kde.score(train_kde_data)
        score_test = kde.score(test_kde_data)
        costs_train[ker].append(score_train)
        costs_test[ker].append(score_test)

for ker in kernels:
    # plt.plot(bws, costs_train[ker], label=f"trian {ker}")
    plt.plot(bws, costs_test[ker], label=f"test {ker}")

plt.legend()
plt.show()




exit()
# density estimation using kde
# for moi_i in range(1, 13):
#     mask_moi_i = tracks[mask]["moi"] == moi_i
#     moi_i_tracks = tracks[mask][mask_moi_i]
#     kde_data = []
#     for i , row in moi_i_tracks.iterrows():
#         for p in row['trajectory']:
#             kde_data.append(p)
#     kde_data = np.array(kde_data)
#     kde = KDE(bandwidth=10)
#     kde.fit(kde_data)
#     scores_cord = kde.score_samples(Data_cord).reshape(X_cord.shape)
#     plt.imshow(img)
#     plt.contourf(X_cord, Y_cord, scores_cord, alpha = 0.2)
#     plt.show()

# exit()

# for k  in range(4):
#     for j in range(4):
#         if k == j: continue
#         img1 = cv.imread(top_image)
#         kde_data = []
#         for i, row in tqdm(tracks[mask].iterrows(), total=len(tracks[mask])):
#             traj = row["trajectory"]
#             for p in traj:
#                 kde_data.append(p)
#             i_str = row["i_str"]
#             i_end = row["i_end"]
#             if i_str == k and i_end == j:
#                 c=0
#                 for x, y in traj:
#                     x, y = int(x), int(y)
#                     img1 = cv.circle(img1, (x,y), radius=1, color=(int(c/len(traj)*255), 70, int(255 - c/len(traj)*255)), thickness=2)
#                     c+=1

#         kde_data = np.array(kde_data)
#         kde = KDE(bandwidth=10)
#         kde.fit(kde_data)
#         scores_cord = kde.score_samples(Data_cord).reshape(X_cord.shape)
#         plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
#         plt.contourf(X_cord, Y_cord, scores_cord, alpha = 0.2)
#         plt.show()

## cluster based on starting positions
# plt.hist(tracks[mask]['moi'])
# plt.show()

starts = []
ends=[]
for i, row in tracks[mask].iterrows():
    starts.append(row['trajectory'][0])
    ends.append(row['trajectory'][-1])

starts = np.array(starts)
ends = np.array(ends)
plt.imshow(img)
plt.scatter(starts[:, 0],starts[:, 1], alpha=0.3, color='green')
plt.scatter(ends[:, 0],ends[:, 1], alpha=0.3, color='red')
plt.show()


# for mi in range(1, 13):
#     tracks_mi = tracks[mask][tracks[mask]['moi']==mi]
#     index_mi = tracks_mi.index
#     starts = []
#     ends=[]
#     for i, row in tracks_mi.iterrows():
#         starts.append(row['trajectory'][0])
#         ends.append(row['trajectory'][-1])
#     starts = np.array(starts)
#     ends = np.array(ends)
#     num_cluster = moi_clusters[mi]
#     # gmm_start = GaussianMixture(n_components=num_cluster, covariance_type='full')
#     # gmm_start.fit(starts)
#     # c_start = gmm_start.predict(starts)
#     # p_start = gmm_start.predict_proba(starts)
#     # cluster_labels = np.unique(c_start)
#     clt = KMeans(n_clusters = num_cluster)
#     clt.fit(starts)
#     c_starts = clt.predict(starts)
#     plt.imshow(img)
#     plt.scatter(starts[:, 0], starts[:, 1], c = c_starts)
#     plt.show()


chosen_indexes = []
for mi in range(1, 13):
    tracks_mi = tracks[mask][tracks[mask]['moi']==mi]
    index_mi = tracks_mi.index
    starts = []
    ends=[]
    for i, row in tracks_mi.iterrows():
        starts.append(row['trajectory'][0])
        ends.append(row['trajectory'][-1])
    starts = np.array(starts)
    ends = np.array(ends)
    num_cluster = moi_clusters[mi]
    if len(starts) < 1:
        continue

    clt_data = np.concatenate([starts, ends], axis=-1) 
    clt = KMeans(n_clusters = num_cluster)
    clt.fit(clt_data)
    c_start = clt.predict(clt_data)
    p_start = np.array([clt.score(x.reshape(1, -1)) for x in clt_data])
    cluster_labels = np.unique(c_start)
    plt.imshow(img)
    plt.scatter(starts[:, 0], starts[:, 1], c=c_start)
    plt.scatter(ends[:, 0], ends[:, 1], c=c_start)
    plt.show()
    for c_label in cluster_labels:
        mask_c = c_start == c_label
        i_c = np.argmax(p_start[mask_c])
        chosen_index = index_mi[mask_c][i_c]
        chosen_indexes.append(chosen_index)

img1 = cv.imread(top_image)
for i, row in tracks.loc[chosen_indexes].iterrows():
    traj = row["trajectory"]
    c=0
    for x, y in traj:
        x, y = int(x), int(y)
        img1 = cv.circle(img1, (x,y), radius=1, color=(int(c/len(traj)*255), 70, int(255 - c/len(traj)*255)), thickness=1)
        c+=1

plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
plt.show()



exit()



features = []
for i, row in tracks.iterrows():
    try:
        f = fit_circle(row['trajectory'])
        if f[-1] < 500:
            features.append(f)
    except: pass
    # x, y, r = features[-1]
    # figure, axes = plt.subplots() 
    # plt.imshow(img)
    # cc = plt.Circle((x, y), r, fill=False, color='red')
    # plt.scatter(x, y, color='red')
    # axes.set_aspect( 1 ) 
    # axes.add_artist( cc )
    # for x, y in row['trajectory']:
    #     plt.scatter(x, y, color='black')
    # plt.show()

plt.imshow(img)
plt.scatter([x for x, _, _ in features], [y for _, y, _ in features], c=[r for _, _, r in features])
plt.show()



# arc_lens = []
# for i, row in tracks_meter.iterrows():
#     arc_lens.append(arc_length(row['trajectory']))
# arc_lens = np.array(arc_lens)
# mask = arc_lens > 8



# gmm_start = GaussianMixture(n_components=4)
# gmm_start.fit(starts)
# p_start = gmm_start.predict_proba(starts)
# p_start = np.max(p_start, axis=1)
# mask = p_start < 0.97
# # clt = DBSCAN(eps=10)
# # p_start = clt.fit_predict(starts)


# # plt.imshow(img)
# # plt.scatter(ends[:, 0],ends[:, 1], alpha=0.2, color='r')
# # plt.show()
