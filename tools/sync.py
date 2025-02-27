import numpy as np
from glob import glob
from dtaidistance import dtw
import matplotlib.pyplot as plt


def sync_NI_PY_times(
    isoforce_iso, isoforce_py, seg_idx: int = 0, plotting: bool = True
):
    tmp_iso_seg = isoforce_iso.torque_segments[f"T_seg_{seg_idx}"]
    tmp_py_seg_raw = isoforce_py.torque_raw_segments[f"T_seg_raw_{seg_idx}"]
    tmp_ts_seg = isoforce_py.timestamp_segments[f"ts_seg_{seg_idx}"]
    assert len(tmp_ts_seg) == len(tmp_py_seg_raw)

    indices = np.linspace(0, len(tmp_iso_seg) - 1, len(tmp_ts_seg), dtype=int)
    indices_py_seg = np.linspace(0, len(tmp_iso_seg) - 1, len(tmp_ts_seg), dtype=int)
    assert len(indices) == len(tmp_py_seg_raw) == len(tmp_ts_seg)
    sampled_iso = tmp_iso_seg[indices]

    if plotting:
        plt.figure(figsize=(6, 2))
        plt.title(f"Segment index {seg_idx}")
        plt.plot(tmp_iso_seg, label="NI-iso")
        plt.plot(
            indices_py_seg, tmp_py_seg_raw * 60, "--", c="C4", label="PY-iso (approx)"
        )
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


def sync_eit_ISO_segments(
    part_path, isoforce_iso, isoforce_py, mode="fast", plotting: bool = True
):
    eit_start_ts = 0
    EIT = list()
    TORQUE = list()
    TS_iso = list()
    TS_eit = list()

    print("Loading eit...")
    times_eit, eit = load_eit_npz(part_path)

    print("Matching sequences...")
    for seg_idx in range(len(isoforce_iso.torque_segments.keys())):
        tmp_ts_seg, sampled_iso = sync_NI_PY_times(
            isoforce_iso, isoforce_py, seg_idx, plotting
        )

        if mode == "fast":

            sync_lst = list()
            for ts in tmp_ts_seg:
                sync_lst.append(find_closest_index(times_eit, ts))
            sync_lst = np.array(sync_lst)

            # eit_start_ts = find_closest_index(times_eit, tmp_ts_seg[0])
            # eit_stop_ts = find_closest_index(times_eit, tmp_ts_seg[-1])

            print(f"detected start idx values{sync_lst[0]=}, {times_eit[sync_lst[0]]=}")

        elif mode == "slow":
            eit_start_ts, eit_start_distance = find_best_match(
                times_eit, tmp_ts_seg[:5]
            )
            print(
                f"Best match at index: {eit_start_ts}, DTW-distanz: {eit_start_distance}"
            )

        print(
            sync_lst[0],
            "diff start:",
            tmp_ts_seg[0] - times_eit[sync_lst[0]],
            "diff end:",
            tmp_ts_seg[-1] - times_eit[sync_lst[-1]],
        )
        assert len(times_eit[sync_lst]) == len(tmp_ts_seg)

        # corresponding eit sequence:
        if mode == "fast":
            eit_sync_seq = eit[sync_lst]
            eit_ts_seq = times_eit[sync_lst]

        elif mode == "slow":
            eit_sync_seq = eit[eit_start_ts : eit_start_ts + len(tmp_ts_seg)]
            eit_ts_seq = times_eit[eit_start_ts : eit_start_ts + len(tmp_ts_seg)]
        assert (
            len(eit_sync_seq) == len(tmp_ts_seg) == len(sampled_iso) == len(eit_ts_seq)
        )

        # plt.figure(figsize=(6, 2))
        # plt.title("err")
        # plt.plot(times_eit[eit_start_ts:eit_start_ts+len(tmp_ts_seg)]-tmp_ts_seg)
        # plt.show()

        EIT.append(eit_sync_seq)
        TORQUE.append(sampled_iso)
        TS_iso.append(tmp_ts_seg)
        TS_eit.append(eit_ts_seq)

    EIT = np.concatenate(EIT)
    TORQUE = np.concatenate(TORQUE)
    TS_iso = np.concatenate(TS_iso)
    TS_eit = np.concatenate(TS_eit)

    return EIT, TORQUE, TS_iso, TS_eit
