import numpy as np
from os.path import join
from glob import glob


def z_score(data, print_info: bool = True):
    if print_info:
        print("before:", np.mean(data), np.std(data))

    mean = np.mean(data)
    std = np.std(data)
    data_z = (data - mean) / std

    if print_info:
        print("after global:", np.mean(data_z), np.std(data_z))

    return data_z


def load_data(
    P_nums: list,
    z_score_norm: str = "global",
    path: str = "data/prepared_data",
    print_info: bool = True,
):
    """
    z_score ... normalization
        - global
        - participant
        - participant_meanfree
    """
    P_str = ["P{0:02d}".format(p) for p in P_nums]
    if print_info:
        print("load:", P_str)
    X = list()
    Y = list()
    P = list()

    if z_score_norm == "global":
        for Pn, Ps in zip(P_nums, P_str):
            l_path = join(path, Ps)
            for ele in np.sort(glob(join(l_path, "*.npz"))):
                tmp = np.load(ele, allow_pickle=True)
                X.append(tmp["EIT"])
                Y.append(tmp["TORQUE"])
                P.append(Pn)

        X = np.abs(X)
        X = z_score(X, print_info)
        Y = np.array(Y)
        P = np.array(P)

    elif z_score_norm == "participant":
        for Pn, Ps in zip(P_nums, P_str):
            Xs = list()
            l_path = join(path, Ps)
            for ele in np.sort(glob(join(l_path, "*.npz"))):
                tmp = np.load(ele, allow_pickle=True)
                Xs.append(tmp["EIT"])
                Y.append(tmp["TORQUE"])
                P.append(Pn)
            Xs = np.abs(Xs)
            Xs = z_score(Xs, print_info)
            X.append(Xs)
        X = np.concatenate(X)
        Y = np.array(Y)
        P = np.array(P)

    elif z_score_norm == "participant_meanfree":
        for Pn, Ps in zip(P_nums, P_str):
            Xs = list()
            l_path = join(path, Ps)
            for ele in np.sort(glob(join(l_path, "*.npz"))):
                tmp = np.load(ele, allow_pickle=True)
                Xs.append(tmp["EIT"])
                Y.append(tmp["TORQUE"])
                P.append(Pn)
            Xs_mean = np.mean(Xs, axis=(0))
            Xs = Xs - Xs_mean
            Xs = np.abs(Xs)
            Xs = z_score(Xs, print_info)
            X.append(Xs)
        X = np.concatenate(X)
        Y = np.array(Y)
        P = np.array(P)

    return X, Y, P
