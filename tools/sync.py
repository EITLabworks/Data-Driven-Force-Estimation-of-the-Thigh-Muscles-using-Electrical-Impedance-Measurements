import numpy as np
from glob import glob
from dtaidistance import dtw
import matplotlib.pyplot as plt


def sync_NI_PY_times(
    isoforce_iso, isoforce_py, seg_idx: int = 1, plotting: bool = True
):
    tmp_iso_seg = isoforce_iso.torque_segments[f"T_seg_{seg_idx}"]
    tmp_py_seg = isoforce_py.torque_segments[f"T_seg_{seg_idx}"]
    tmp_ts_seg = isoforce_py.timestamp_segments[f"ts_seg_{seg_idx}"]

    indices = np.linspace(0, len(tmp_iso_seg) - 1, len(tmp_py_seg), dtype=int)
    assert len(indices) == len(tmp_py_seg)
    sampled_iso = tmp_iso_seg[indices]

    if plotting:
        plt.figure(figsize=(6, 2))
        plt.title(f"Segment index {seg_idx}")
        plt.plot(tmp_iso_seg, label="NI-iso")
        plt.scatter(
            indices, sampled_iso, marker="x", s=20, c="C1", label="EIT timestamps"
        )
        plt.legend()
        # plt.hlines(protocol.IsokinetikMeasurement.force_levels[seg_idx],xmin=0,xmax=max(indices))
        plt.ylabel("torque (NM)")
        plt.xlabel("Segment time indices $k$")
        plt.grid()
        plt.show()
    return tmp_ts_seg, sampled_iso


def load_eit_npz(part_path):
    times_eit = list()
    eit = list()
    for ele in np.sort(glob(part_path.s_path_eit + "*.npz")):
        tmp = np.load(ele, allow_pickle=True)
        times_eit.append(tmp["timestamp"])
        eit.append(tmp["eit"])
    times_eit = np.array(times_eit)
    eit = np.array(eit)
    return times_eit, eit


def find_closest_index(array, value):
    """Used for time sync between eit and isoforce"""
    index = np.argmin(np.abs(array - value))
    return index


def find_best_match(long_series, reference, window_size=4):
    best_index = None
    best_distance = float("inf")

    for i in range(len(long_series) - window_size + 1):
        segment = long_series[i : i + window_size]
        distance = dtw.distance(reference, segment)

        if distance < best_distance:
            best_distance = distance
            best_index = i

    return best_index, best_distance
