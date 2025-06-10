from dataclasses import dataclass
from glob import glob
from os.path import join
from typing import Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json
import os


import numpy as np

from .postprocessing import (
    extract_timestamp_and_sample,
    conv_array_float,
    edge_detection,
    scale_to_range,
    lowpass_filter,
)


@dataclass
class Participant:
    Number: str
    age: str
    gender: str
    leg: str


@dataclass
class IsokinetikMeasurement:
    rotation_velocity: str
    force_levels: np.ndarray


@dataclass
class EITmeasurement:
    excitation_frequency: Union[int, float]
    burst_count: int
    amplitude: Union[int, float]
    frame_rate: int
    n_el: int
    injection_skip: int


class Protocol:
    def __init__(self, path: str, prints: bool = True):
        self.path = path
        self.prints = prints

        self.read_json()

    def read_json(self):
        self.json_path = glob(join(self.path, "*protocol.json"))[0]
        with open(self.json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if self.prints:
            print(self.json_path)
            print(data)

        # Participant
        self.Participant = Participant(**data["participant"])

        # IsokinetikMeasurement
        rotation_velocity = int(
            data["isokinetic_measurement"]["rotation_velocity"].split(" ")[0]
        )
        force_levels = data["isokinetic_measurement"]["force_levels"]
        force_levels = np.array(
            list(map(int, force_levels.strip("[]").split())), dtype=int
        )
        self.IsokinetikMeasurement = IsokinetikMeasurement(
            rotation_velocity, force_levels
        )
        # EITmeasurement
        excitation_frequency = data["eit_measurement"]["excitation_frequency"]
        burst_count = data["eit_measurement"]["burst_count"]
        amplitude = int(data["eit_measurement"]["amplitude"].split(" ")[0])
        frame_rate = data["eit_measurement"]["frame_rate"]
        n_el = data["eit_measurement"]["n_el"]
        injection_skip = data["eit_measurement"]["injection_skip"]

        self.EITmeasurement = EITmeasurement(
            excitation_frequency,
            burst_count,
            amplitude,
            frame_rate,
            n_el,
            injection_skip,
        )

        # Notes
        self.notes = data["notes"]


### Class for the directory which is post-processed


class ProcessingDir:
    def __init__(self, path):
        self.path = path
        self.get_paths()
        self.print_info()

    def get_paths(self):
        self.isoforce_iso = glob(self.path + "*.txt")[0]
        self.isoforce_py_raw = join(self.path, "iso_raw/")
        self.sciospec_EIT_raw = join(self.path, "eit_raw/")
        self.EIT_samples_raw = glob(self.sciospec_EIT_raw + "2025*/setup/")[0]
        self.s_path_eit = join(self.path, "EIT_processed/")

    def print_info(self):
        print("Fund participant data:\n")
        print(f"Head directory: {self.path=}")
        print(f"Raw Isoforce data measured by Isoforce:\n\t{self.isoforce_iso=}")
        print(f"Raw Isoforce data measured by Python:\n\t{self.isoforce_py_raw=}")
        print(f"Raw sciospec EIT data:\n\t{self.sciospec_EIT_raw=}")
        print(f"Raw sciospec EIT samples:\n\t{self.EIT_samples_raw=}")
        print(f"Preprocessed sciospec EIT samples:\n\t{self.s_path_eit=}")


### Isoforce Class


class IsoforceIso:
    def __init__(self, DF, protocol: Protocol, LP_filter=False):
        self.DF = DF
        self.protocol = protocol
        self.LP_filter = LP_filter

        self.init_data()
        self.detect_start_stop_idxs()
        self.export_segments()
        self.filter_torque()

    def init_data(self):
        # read raw torque data

        if self.protocol.Participant.leg == "left":
            # self.angle = -1 * conv_array_float(self.DF["Angle"])
            # self.speed = -1 * conv_array_float(self.DF["Velocity"])
            torque_raw = -1 * conv_array_float(self.DF["Torque"])
        elif self.protocol.Participant.leg == "right":
            torque_raw = conv_array_float(self.DF["Torque"])

        self.angle = conv_array_float(self.DF["Angle"])
        self.speed = conv_array_float(self.DF["Velocity"])
        if self.LP_filter:
            print("!!!The torque data is lowpass filtered!!!")
            self.torque_raw = lowpass_filter(torque_raw)
            print("!!!The angle data is lowpass filtered!!!")
            self.angle = lowpass_filter(self.angle, cutoff=2, fs=400)
        else:
            self.torque_raw = torque_raw

        date_part = self.DF[""][5].split("/")[1:]
        time_part = self.DF[""][6].split("/")[1]
        date_time_string = f"{date_part[2]}-{date_part[1]}-{date_part[0]} {time_part}"
        self.date_time_iso = datetime.strptime(date_time_string, "%Y-%m-%d %H:%M:%S")

        # unused:
        # self.record = self.DF["Record"]
        # self.direction = self.DF["Direction"]
        # self.mode = self.DF["Mode"]

    def detect_start_stop_idxs(self):
        # use the speed edges for start and stop times
        k = np.arange(len(self.speed))
        dx_dk = np.gradient(self.speed, k)
        self.start_idxs = np.where(dx_dk > np.mean(dx_dk))[0][1::2]
        self.stop_idxs = np.where(dx_dk < -np.mean(dx_dk))[0][::2]
        # length quality check
        to_short = np.where(self.stop_idxs - self.start_idxs < 1000)[0]
        to_long = np.where(self.stop_idxs - self.start_idxs > 2500)[0]
        cut_out = np.concatenate([to_short, to_long])
        self.stop_idxs = np.delete(self.stop_idxs, cut_out)
        self.start_idxs = np.delete(self.start_idxs, cut_out)

    def export_segments(self):
        idx = 0
        T_segment_dict = dict()
        A_segment_dict = dict()
        exclude_window = np.zeros(len(self.speed))
        for start, stop in zip(self.start_idxs, self.stop_idxs):
            T_segment_dict[f"T_seg_{idx}"] = self.torque_raw[start:stop]
            A_segment_dict[f"A_seg_{idx}"] = self.angle[start:stop]
            exclude_window[start:stop] = 1
            idx += 1

        self.exclude_window = exclude_window
        self.angle_segments = A_segment_dict
        self.torque_segments = T_segment_dict

    def filter_torque(self):
        self.torque = self.torque_raw * self.exclude_window

    def plot_angle(self):
        """P -> angle"""
        plt.figure(figsize=(12, 3))
        plt.plot(self.angle, "C3")
        plt.grid()
        plt.xlabel("sample ($k$)")
        plt.ylabel("Angle (°)")
        plt.show()

    def plot_torque(self):
        """T -> torque"""
        tks = np.round(np.linspace(np.min(self.torque_raw), np.max(self.torque_raw), 5))

        plt.figure(figsize=(12, 3))
        plt.plot(self.torque, "C0", label=".torque")
        plt.plot(self.torque_raw, "C8", lw=0.5, label=".torque_raw")
        plt.grid()
        plt.legend()
        plt.xlabel("sample ($k$)")
        plt.ylabel("Torque (NM)")
        plt.show()

    def plot_speed(self):
        """S -> speed"""
        plt.figure(figsize=(12, 3))
        plt.plot(self.speed, "C8")
        plt.scatter(
            self.start_idxs, self.speed[self.start_idxs], c="C2", label="start idx"
        )
        plt.scatter(
            self.stop_idxs, self.speed[self.stop_idxs], c="C3", label="stop idxs"
        )
        plt.grid()
        plt.xlabel("sample ($k$)")
        plt.ylabel("Speed (°/s)")
        plt.legend(loc="upper left")
        plt.show()

    def plot_data(self, filename=None):
        plt.figure(figsize=(12, 3))
        plt.plot(self.angle, "C3", label="Angle")
        plt.plot(self.torque, "C0", label="Torque")
        plt.plot(self.speed, "C8", label="Speed")
        plt.scatter(
            self.start_idxs, self.speed[self.start_idxs], c="C2", label="start idx"
        )
        plt.scatter(
            self.stop_idxs, self.speed[self.stop_idxs], c="C4", label="stop idxs"
        )
        plt.legend(loc="upper left")
        plt.grid()
        plt.xlabel("sample ($k$)")
        if filename != None:
            plt.tight_layout()
            plt.savefig(filename)
        plt.show()


class IsoforcePy:
    def __init__(
        self,
        path,
        protocol: Protocol,
        LP_filter=True,
        over_UTC=False,
        scale_0_1=True,
        speed_window_trunc=True,
        segment_len_threshold: int = 200,
        phase_shift=0,
    ):
        """
        path ... part_path.isoforce_py_raw.
        LP_filter ... low-pass filter the torque data.
        over_UTC ... plot over the measured time stamps.
        scale_0_1 ... scale all analog measured values between 0 and 1.
        speed_window_trunc ... create a speed window.
        phase_shift ... time index phase shift between Isoforce and Python (heuristic).
        """
        self.path = path
        self.protocol = protocol
        self.LP_filter = LP_filter
        self.over_UTC = over_UTC
        self.speed_window_trunc = speed_window_trunc
        self.segment_len_threshold = segment_len_threshold
        self.scale_0_1 = scale_0_1
        self.phase_shift = phase_shift

        self.init_data()
        self.export_segments()
        self.filter_torque()

    def init_data(self):
        # Initialize lists to store aggregated data
        angle = list()  # channel 1
        torque = list()  # channel 2
        speed = list()  # channel 3
        time = list()

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
            angle.extend(ch_1)
            torque.extend(ch_2)
            speed.extend(ch_3)
            time.extend(timestamps_expanded)

        if self.protocol.Participant.leg == "left":
            angle = -1 * np.array(angle)
            torque = -1 * np.array(torque)
            speed = -1 * np.array(speed)

        # set class variables
        self.angle = np.array(angle)
        self.torque_raw = np.array(torque)
        if self.LP_filter == True:
            self.torque = lowpass_filter(self.torque_raw)
        else:
            self.torque = self.torque_raw
        self.speed = np.array(speed)

        if self.scale_0_1:
            self.angle = scale_to_range(self.angle)
            self.torque = scale_to_range(self.torque)
            self.torque_raw = scale_to_range(self.torque_raw)
            self.speed = scale_to_range(self.speed)

        if self.speed_window_trunc:
            speed_window = self.speed
            speed_window[self.speed <= 0.95] = 0
            speed_window[self.speed > 0.5] = 1
            self.speed_window = speed_window

        assert len(angle) == len(torque) == len(speed)
        self.time = np.array(time)
        self.timestamps = np.array([dt.timestamp() for dt in self.time])

    def export_segments(
        self,
        distance=500,
        height=0.8,
    ):
        idx = 0
        T_segment_dict = dict()
        T_segment_raw_dict = dict()
        A_segment_dict = dict()
        ts_segment_dict = dict()

        # self.stop_idxs, _ = find_peaks(self.angle, distance=distance, height=height)

        # detect rising and falling edges
        detected_rising = edge_detection(self.speed, mode="rising")
        detected_falling = edge_detection(self.speed, mode="falling")

        if detected_rising[0] > detected_falling[0] and len(detected_rising) != len(
            detected_falling
        ):
            detected_rising = np.insert(detected_rising, 0, 0)

        self.start_idxs = detected_rising
        self.stop_idxs = detected_falling

        # start_filt = list()
        # for stop in self.stop_idxs:
        #    diff = stop - detected_rising
        #    min_diff = np.argmin(diff[diff > 0])
        #    start_filt.append(detected_rising[min_diff])
        # self.start_idxs = np.array(start_filt) - self.phase_shift

        # exclude all segments that are shorter than 300 sample (~3s)
        len_mask = self.stop_idxs - self.start_idxs > self.segment_len_threshold
        self.start_idxs = self.start_idxs[len_mask]
        self.stop_idxs = self.stop_idxs[len_mask]

        assert (
            self.start_idxs.shape == self.stop_idxs.shape
        ), "Error during segment detection."

        exclude_window = np.zeros(len(self.torque))
        for start, stop in zip(self.start_idxs, self.stop_idxs):
            T_segment_dict[f"T_seg_{idx}"] = self.torque[start:stop]
            T_segment_raw_dict[f"T_seg_raw_{idx}"] = self.torque_raw[start:stop]
            A_segment_dict[f"A_seg_{idx}"] = self.angle[start:stop]

            # timestapm segment
            ts_segment = [dt.timestamp() for dt in self.time[start:stop]]
            ts_segment_dict[f"ts_seg_{idx}"] = np.array(ts_segment)

            exclude_window[start:stop] = 1
            idx += 1

        self.angle_segments = A_segment_dict
        self.torque_segments = T_segment_dict
        self.torque_raw_segments = T_segment_raw_dict
        self.timestamp_segments = ts_segment_dict
        self.exclude_window = exclude_window

    def filter_torque(self):
        self.torque = self.torque * self.exclude_window

    def plot_angle(self):
        plt.figure(figsize=(12, 3))
        if self.over_UTC:
            plt.plot(self.time, self.angle, label="Angle", color="C3")
            plt.xlabel("Time (UTC)")
        else:
            plt.plot(self.angle, label="Angle", color="C3")
            plt.xlabel("sample ($k$)")
        plt.ylabel("Angle")
        plt.grid(True)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    def plot_torque(self):
        plt.figure(figsize=(12, 3))
        if self.over_UTC:
            plt.plot(self.time, self.torque, "C0", label=".torque")
            plt.plot(self.time, self.torque_raw, "C5", lw=0.5, label=".torque_raw")
            plt.xlabel("Time (UTC)")
        else:
            plt.plot(self.torque, "C0", label=".torque")
            plt.plot(self.torque_raw, "C5", lw=0.5, label=".torque_raw")
            plt.xlabel("sample ($k$)")
        plt.ylabel("Torque (Nm)")
        plt.grid(True)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    def plot_speed(self):
        plt.figure(figsize=(12, 3))
        if self.over_UTC:
            plt.plot(self.time, self.speed, label="Speed", color="C8")
            plt.xlabel("Time (UTC)")
        else:
            plt.plot(self.speed, label="Speed", color="C8")
            plt.xlabel("sample ($k$)")
        plt.ylabel("Speed (°/s)")
        plt.grid(True)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()
