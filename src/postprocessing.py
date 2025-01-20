import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import os
import re
from datetime import datetime, timedelta, timezone

from sciopy.doteit import convert_fulldir_doteit_to_npz

### Filtering


def lowpass_filter(data, cutoff=2, fs=100, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_signal = filtfilt(b, a, data)
    return filtered_signal


### Classes


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
        plt.plot(self.torque, "C0")
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
        plt.legend(loc="upper left")
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
        plt.legend(loc="upper left")
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


### Process Python Isoforce data


def extract_timestamp_and_sample(filename):
    # Function to extract timestamp and sample number from filename
    match = re.search(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(\d+)\.npz$", filename)
    if match:
        timestamp_str = match.group(1)  # Extract the timestamp
        sample_number = int(match.group(2))  # Extract the sample number
        timestamp = datetime.strptime(
            timestamp_str, "%Y-%m-%d_%H-%M-%S"
        )  # Convert to datetime
        return (timestamp, sample_number)
    return (None, None)


class IsoforcePy:
    def __init__(self, path, LP_filter=True, over_UTC=False):
        """
        path ... part_path.isoforce_py_raw.
        LP_filter ... Low-pass filter the torque data.
        over_UTC ... plot over the measured time stamps.
        """
        self.path = path
        self.LP_filter = LP_filter
        self.over_UTC = over_UTC
        self.init_data()

    def init_data(self):
        # Initialize lists to store aggregated data
        all_position = list()  # channel 1
        all_torque = list()  # channel 2
        all_speed = list()  # channel 3
        all_time = list()

        file_list = sorted(
            [f for f in os.listdir(self.path) if f.endswith(".npz")],
            key=lambda f: (extract_timestamp_and_sample(f)),
        )

        last_file_for_timestamp = dict()

        for file_name in file_list:
            timestamp, sample_number = extract_timestamp_and_sample(file_name)
            last_file_for_timestamp[timestamp] = (
                file_name  # Overwrite with the latest file
            )

        for timestamp, last_file in last_file_for_timestamp.items():
            file_path = os.path.join(self.path, last_file)
            data = np.load(file_path, allow_pickle=True)

            # Extract the data for the last file of this timestamp
            ch_1, ch_2, ch_3 = data["data"]
            assert len(ch_1) == len(ch_2) == len(ch_3)

            timestamps_start = data["timestamps_start"]
            timestamps = data["timestamps_current"]
            sampling_rate = data["sampling_rate"]

            # Expand timestamps for the last sample file
            timestamps_expanded = [
                timestamp + timedelta(seconds=(i / sampling_rate))
                for i in range(len(ch_1))
            ]

            # Append the torque and time data
            all_position.extend(ch_1)
            all_torque.extend(ch_2)
            all_speed.extend(ch_3)
            all_time.extend(timestamps_expanded)

        self.all_position = np.array(all_position)
        if self.LP_filter == True:
            self.all_torque_LP = lowpass_filter(all_torque)
        self.all_torque = np.array(all_torque)
        self.all_speed = np.array(all_speed)

        assert len(all_position) == len(all_torque) == len(all_speed)
        if self.over_UTC == False:
            self.all_time = np.arange(len(all_time))

    def plot_angle(self):
        plt.figure(figsize=(12, 3))
        plt.plot(self.all_time, self.all_position, label="Position", color="C3")
        if self.over_UTC == False:
            plt.xlabel("sample $k$")
        else:
            plt.xlabel("Time (UTC)")
        plt.ylabel("Position")
        plt.grid(True)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    def plot_torque(self):
        plt.figure(figsize=(12, 3))
        if self.LP_filter == True:
            plt.plot(
                self.all_time, self.all_torque, "--", label="Torque raw", color="C0"
            )
            plt.plot(self.all_time, self.all_torque_LP, label="Torque LP", color="C9")
        else:
            plt.plot(self.all_time, self.all_torque, label="Torque raw", color="C0")
        if self.over_UTC == False:
            plt.xlabel("sample $k$")
        else:
            plt.xlabel("Time (UTC)")
        plt.ylabel("Torque (Nm)")
        plt.grid(True)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    def plot_velocity(self):
        plt.figure(figsize=(12, 3))
        plt.plot(self.all_time, self.all_speed, label="Speed", color="C8")
        if self.over_UTC == False:
            plt.xlabel("sample $k$")
        else:
            plt.xlabel("Time (UTC)")
        plt.ylabel("Speed (°/s)")
        plt.grid(True)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()
