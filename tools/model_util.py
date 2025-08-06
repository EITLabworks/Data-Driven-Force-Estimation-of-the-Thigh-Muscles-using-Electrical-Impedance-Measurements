import numpy as np
from os.path import join
from glob import glob

lever_arm = {  # cm
    "P01": 0.364,
    "P02": 0.29,
    "P03": 0.293,
    "P04": 0.284,
    "P05": 0.309,
    "P06": 0.314,
    "P07": 0.354,
    "P08": 0.304,
    "P09": 0.319,
    "P10": 0.28,
    "P11": 0.341,
    "P12": 0.318,
    "P13": 0.30,
    "P14": 0.287,
    "P15": 0.294,
}


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

    returns:
    X ... EIT
    F ... Force
    Y ... Torque
    P ... Participant
    """

    P_str = ["P{0:02d}".format(p) for p in P_nums]
    if print_info:
        print("load:", P_str)
    X = list()
    Y = list()  # Torque
    F = list()  # Force
    P = list()

    if z_score_norm == "global":
        for Pn, Ps in zip(P_nums, P_str):
            l_path = join(path, Ps)
            for ele in np.sort(glob(join(l_path, "*.npz"))):
                tmp = np.load(ele, allow_pickle=True)
                X.append(tmp["EIT"])
                Y.append(tmp["TORQUE"])
                F.append(tmp["TORQUE"] / lever_arm[Ps])
                P.append(Pn)

        X = np.abs(X)
        X = z_score(X, print_info)
        Y = np.array(Y)
        F = np.array(F)
        P = np.array(P)

    elif z_score_norm == "participant":
        for Pn, Ps in zip(P_nums, P_str):
            Xs = list()
            l_path = join(path, Ps)
            for ele in np.sort(glob(join(l_path, "*.npz"))):
                tmp = np.load(ele, allow_pickle=True)
                Xs.append(tmp["EIT"])
                Y.append(tmp["TORQUE"])
                F.append(tmp["TORQUE"] / lever_arm[Ps])
                P.append(Pn)
            Xs = np.abs(Xs)
            Xs = z_score(Xs, print_info)
            X.append(Xs)
        X = np.concatenate(X)
        Y = np.array(Y)
        F = np.array(F)
        P = np.array(P)

    elif z_score_norm == "participant_meanfree":
        for Pn, Ps in zip(P_nums, P_str):
            Xs = list()
            l_path = join(path, Ps)
            for ele in np.sort(glob(join(l_path, "*.npz"))):
                tmp = np.load(ele, allow_pickle=True)
                Xs.append(tmp["EIT"])
                Y.append(tmp["TORQUE"])
                F.append(tmp["TORQUE"] / lever_arm[Ps])
                P.append(Pn)
            Xs_mean = np.mean(Xs, axis=(0))
            Xs = Xs - Xs_mean
            Xs = np.abs(Xs)
            Xs = z_score(Xs, print_info)
            X.append(Xs)
        X = np.concatenate(X)
        Y = np.array(Y)
        F = np.array(F)
        P = np.array(P)

    return X, F, Y, P
