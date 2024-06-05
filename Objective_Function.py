import numpy as np
from onedal.neighbors import NearestNeighbors
from Global_Vars import Global_Vars
from Model_HCARDNet import Model_HCARDNet


def relief_score(X, num_neighbors=5):
    n_samples, n_features = X.shape
    relief_scores = np.zeros(n_features)
    nn = NearestNeighbors(n_neighbors=num_neighbors + 1).fit(X)

    for i in range(n_samples):
        instance = X[i]
        _, nearest_indices = nn.kneighbors([instance])
        nearest_indices = nearest_indices.flatten()[1:]  # Exclude the instance itself

        near_hit_diff = np.abs(instance - X[nearest_indices]).mean(axis=0)
        far_diff = np.abs(instance - X).max(axis=0)

        relief_scores -= near_hit_diff / num_neighbors
        relief_scores += far_diff / num_neighbors

    return relief_scores


def Objfun(Soln):
    Feat_1 = Global_Vars.Feat_1
    Feat_2 = Global_Vars.Feat_2
    Feat_3 = Global_Vars.Feat_3
    Feat_4 = Global_Vars.Feat_4
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        selected_feat1 = Feat_1[:, np.round(sol[:5]).astype('int')]
        selected_feat2 = Feat_2[:, np.round(sol[5:10]).astype('int')]
        selected_feat3 = Feat_3[:, np.round(sol[10:15]).astype('int')]
        selected_feat4 = Feat_4[:, np.round(sol[15:20]).astype('int')]
        Feature = np.concatenate((selected_feat1, selected_feat2, selected_feat3, selected_feat4), axis=1)
        relif = relief_score(Feature, num_neighbors=5)
        Fitn[i] = 1 / relif
    return Fitn


def Objfun_Cls(Soln):
    Feat = Global_Vars.Feat
    Video = Global_Vars.Video
    Target = Global_Vars.Target
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        Eval = Model_HCARDNet(Feat, Video, Target, sol)
        Fitn[i] = (1 / Eval[4] + Eval[7])
    return Fitn
