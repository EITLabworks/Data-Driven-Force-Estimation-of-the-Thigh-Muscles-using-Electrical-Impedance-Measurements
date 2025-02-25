import pandas as pd
from glob import glob
import numpy as np
from scipy.signal import butter, filtfilt
import re
from datetime import datetime
from tqdm import tqdm

from sciopy.doteit import convert_fulldir_doteit_to_npz

### Filtering - LP filter


def lowpass_filter(data, cutoff=2, fs=100, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_signal = filtfilt(b, a, data)
    return filtered_signal


### Scaling


def scale_to_range(values, new_min=0, new_max=1):
    old_min = np.min(values)
    old_max = np.max(values)

    def scale(value):
        return new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)

    if isinstance(values, (list, tuple)):
        return type(values)(scale(value) for value in values)
    else:
        return scale(values)


### Edge detection


def edge_detection(signal, mode="rising", threshold=1):
    """
    mode = "rising" ... rising edge detection.
    mode = "falling" ... falling edge detection.
    """
    signal = np.asarray(signal)
    diff = np.diff(signal)
    if mode == "rising":
        rising_edges = np.where(diff == threshold)[0]
    elif mode == "falling":
        rising_edges = np.where(diff == -threshold)[0]
    return rising_edges


### Process raw Isoforce data recorded with the Isoforce


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


def conv_array_float(arr):
    return np.array([float(x.replace(",", ".")) for x in arr])


### Python data processing


def process_sciospec_eit(part_path, protocol):

    convert_fulldir_doteit_to_npz(part_path.EIT_samples_raw, part_path.s_path_eit)

    # convert_fulldir_doteit_to_npz(, )
    skip = protocol.EITmeasurement.injection_skip
    n_el = protocol.EITmeasurement.n_el

    for ele in tqdm(np.sort(glob(part_path.s_path_eit + "*.npz"))):
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
