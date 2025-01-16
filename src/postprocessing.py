import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

from sciopy.doteit import convert_fulldir_doteit_to_npz


def renderDF(filename):
    DF = pd.read_csv(filename, delimiter="\t", header=0, skip_blank_lines=True)
    DF = pd.DataFrame(DF)

    for rnms in DF.columns:
        DF.rename(columns={rnms: rnms.split(" ")[0]}, inplace=True)

    DF = DF.loc[:, ~DF.columns.str.contains("^Unnamed")]
    DF = DF.dropna(how="all")
    return DF


def convert_timestamp(date_str):
    if len(str(date_str).split(".")) > 2:
        timestamp = datetime.strptime(date_str, "%Y.%m.%d. %H:%M:%S.%f")
        return timestamp.timestamp()
    else:
        date_time = datetime.fromtimestamp(float(date_str))
        return date_time.strftime("%Y.%m.%d. %H:%M:%S.%f")


class ProcessingDir:
    def __init__(self, path):
        self.path = path
        self.get_paths()
        self.print_info()

    def get_paths(self):
        self.isoforce_iso = glob(self.path + "*.txt")[0]
        self.isoforce_py_raw = self.path + "Isokinetic_raw/"
        self.sciospec_EIT_raw = self.path + "EIT_raw/"
        self.EIT_samples_raw = glob(self.sciospec_EIT_raw + "2025*/setup/")[0]
        self.s_path_eit = self.path + "EIT_processed/"

    def print_info(self):
        print("Fund participant data:\n")
        print(f"Head directory: {self.path=}")
        print(f"Raw Isoforce data measured by Isoforce:\n\t{self.isoforce_iso=}")
        print(f"Raw Isoforce data measured by Python:\n\t{self.isoforce_py_raw=}")
        print(f"Raw sciospec EIT data:\n\t{self.sciospec_EIT_raw=}")
        print(f"Raw sciospec EIT samples:\n\t{self.EIT_samples_raw=}")
        print(f"Preprocessed sciospec EIT samples:\n\t{self.s_path_eit=}")


def conv_array_float(arr):
    return np.array([float(x.replace(",", ".")) for x in arr])


class IsoforceIso:
    def __init__(self, DF):
        self.torque = conv_array_float(DF["Torque"])
        self.angle = conv_array_float(DF["Angle"])
        self.direction = DF["Direction"]
        self.mode = DF["Mode"]
        self.velocity = conv_array_float(DF["Velocity"])
        self.record = DF["Record"]

        self.detect_start_stop_idxs()
        self.export_torque_segments()

    def detect_start_stop_idxs(self):
        # use the velocity edges for start and stop times
        k = np.arange(len(self.velocity))
        dx_dk = np.gradient(self.velocity, k)
        self.start_idxs = np.where(dx_dk > np.mean(dx_dk))[0][1::2]
        self.stop_idxs = np.where(dx_dk < -np.mean(dx_dk))[0][::2]

    def export_torque_segments(self):
        idx = 0
        segment_dict = dict()
        for start, stop in zip(self.start_idxs, self.stop_idxs):
            segment_dict[f"seg_{idx}"] = self.torque[start:stop]
            idx += 1

        self.torque_segments = segment_dict

    def plot_torque(self):
        tks = np.round(np.linspace(np.min(self.torque), np.max(self.torque), 5))

        plt.figure(figsize=(12, 3))
        plt.plot(self.torque)
        plt.grid()
        # plt.yticks(ticks=tks, labels=tks)
        plt.xlabel("sample $k$")
        plt.ylabel("Torque (NM)")
        plt.show()

    def plot_angle(self):
        plt.figure(figsize=(12, 3))
        plt.plot(self.angle)
        plt.grid()
        plt.xlabel("sample $k$")
        plt.ylabel("Angle")
        plt.show()

    def plot_velocity(self):
        plt.figure(figsize=(12, 3))
        plt.plot(self.velocity)
        plt.scatter(
            self.start_idxs, self.velocity[self.start_idxs], c="C2", label="start idx"
        )
        plt.scatter(
            self.stop_idxs, self.velocity[self.stop_idxs], c="C3", label="stop idxs"
        )
        plt.grid()
        plt.xlabel("sample $k$")
        plt.ylabel("Velocity")
        plt.legend()
        plt.show()

    def plot_data(self, filename=None):
        plt.figure(figsize=(12, 3))
        plt.plot(self.torque, label="Torque")
        plt.plot(self.angle / 2, label="Angle / 2")
        plt.plot(self.velocity, label="Velocity")
        plt.scatter(
            self.start_idxs, self.velocity[self.start_idxs], c="C2", label="start idx"
        )
        plt.scatter(
            self.stop_idxs, self.velocity[self.stop_idxs], c="C3", label="stop idxs"
        )
        plt.legend()
        plt.grid()
        plt.xlabel("sample $k$")
        if filename != None:
            plt.tight_layout()
            plt.savefig(filename)
        plt.show()


def process_sciospec_eit(part_path):

    convert_fulldir_doteit_to_npz(part_path.EIT_samples_raw, part_path.s_path_eit)

    # convert_fulldir_doteit_to_npz(, )
    skip = 5
    n_el = 16

    for ele in glob(part_path.s_path_eit + "*.npz"):
        tmp_eit = np.load(ele, allow_pickle=True)

        els = np.arange(1, n_el + 1)
        mat = np.zeros((n_el, n_el), dtype=complex)

        for i1, i2 in zip(els, np.roll(els, -(skip + 1))):
            mat[i1 - 1, :n_el] = tmp_eit[f"{i1}_{i2}"][:n_el]

        np.savez(
            ele,
            eit=mat,
            timestamp=convert_timestamp(tmp_eit["date_time"].tolist()),
        )
