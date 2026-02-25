from scapy.all import PcapReader, RadioTap, Dot11
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def convert_to_df(PATH):
    rows = []
    total = 0
    kept = 0

    with PcapReader(PATH) as pcap:
        for pkt in pcap:
            total += 1

            # Need 802.11 header
            if not pkt.haslayer(Dot11):
                continue

            dot11 = pkt[Dot11]
            kept += 1

            # Timestamp: scapy stores it as pkt.time (epoch seconds as float)
            t = float(getattr(pkt, "time", None))

            # Frame length in bytes
            frame_len = len(pkt)

            # 802.11 frame control fields
            fc_type = int(dot11.type)        # 0 mgmt, 1 ctrl, 2 data
            fc_subtype = int(dot11.subtype)

            # Addresses (may be None depending on frame type)
            sa = getattr(dot11, "addr2", None)
            da = getattr(dot11, "addr1", None)

            # Duration/ID field
            duration = int(getattr(dot11, "ID", None)) if getattr(dot11, "ID", None) is not None else None

            # Radiotap RSSI/noise (often not present; driver-dependent)
            rssi = None
            noise = None
            if pkt.haslayer(RadioTap):
                rt = pkt[RadioTap]
                # Different drivers populate different attributes; try common ones.
                rssi = getattr(rt, "dBm_AntSignal", None)
                noise = getattr(rt, "dBm_AntNoise", None)

                # Convert to float if present
                rssi = float(rssi) if rssi is not None else None
                noise = float(noise) if noise is not None else None

            rows.append({
                "time_epoch": t,
                "frame_len": frame_len,
                "fc_type": fc_type,
                "fc_subtype": fc_subtype,
                "duration": duration,
                "sa": sa,
                "da": da,
                "rssi_dbm": rssi,
                "noise_dbm": noise,
            })
            

    df = pd.DataFrame(rows)
    normalizer = df["time_epoch"][0]
    df["time_epoch"] -= normalizer
    return df


def bin_frames(df: pd.DataFrame,
               bin_size: float = 0.25,
               time_col: str = "time_epoch",
               max_time: float | None = None,
               include_empty: bool = True) -> pd.DataFrame:
   
    if time_col not in df.columns:
        raise ValueError(f"Expected column '{time_col}' in df")

    out = df.copy()

    # Ensure numeric time
    out[time_col] = pd.to_numeric(out[time_col], errors="coerce")

    # Drop NaN times
    out = out.dropna(subset=[time_col])

    # Optionally clip to [0, max_time)
    if max_time is not None:
        out = out[(out[time_col] >= 0) & (out[time_col] < max_time)].copy()

    # Compute bin index: floor(t / bin_size)
    # Add a tiny epsilon to avoid edge cases like t=0.5 being stored as 0.4999999998
    eps = 1e-9
    out["bin_id"] = np.floor((out[time_col] + eps) / bin_size).astype(int)

    # Bin edges
    out["bin_start"] = out["bin_id"] * bin_size
    out["bin_end"] = out["bin_start"] + bin_size

    if include_empty:
        # Build a complete bin index from 0..last_bin and reindex later if you want.
        # Here we just ensure bin_id is consistent; empty-bin creation is usually done
        # during aggregation, not at the row level.
        pass

    return out


def aggregate_bins(df_binned: pd.DataFrame,
                   bin_size: float = 0.25,
                   bin_col: str = "bin_id") -> pd.DataFrame:
    """
    Example aggregator: counts and bytes per bin (you can extend this later).
    Returns one row per bin_id.
    """
    g = df_binned.groupby(bin_col, sort=True)

    agg = g.agg(
        n_frames=("frame_len", "size"),
        bytes_total=("frame_len", "sum"),
        n_data=("fc_type", lambda s: (s == 2).sum()),
        bytes_data=("frame_len", lambda s: s[df_binned.loc[s.index, "fc_type"] == 2].sum()),
        n_nond=("fc_type", lambda s: (s.isin([0, 1])).sum()),
        bytes_nond=("frame_len", lambda s: s[df_binned.loc[s.index, "fc_type"].isin([0,1])].sum()),
    ).reset_index()

    # Add time edges for each bin
    agg["bin_start"] = agg[bin_col] * bin_size
    agg["bin_end"] = agg["bin_start"] + bin_size
    return agg


import numpy as np
import pandas as pd

def vectorize(
    binned: pd.DataFrame,
    bin_col: str = "bin_id",
    label_col: str = "label",
    return_y: bool = True,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Returns:
      X: (n_bins, 9) features
      y: (n_bins,) labels (same label repeated for each bin)
    """
    if bin_col not in binned.columns:
        raise ValueError(f"Expected bin column '{bin_col}' in dataframe")
    if label_col not in binned.columns:
        raise ValueError(f"Expected label column '{label_col}' in dataframe")

    if binned.empty:
        X = np.zeros((0, 9), dtype=float)
        y = np.zeros((0,), dtype=int)
        return (X, y) if return_y else X

    # get a single label for the whole capture
    labels = binned[label_col].dropna().unique()
    if len(labels) != 1:
        raise ValueError(f"Expected exactly one unique label in '{label_col}', got {labels}")
    label = int(labels[0])

    max_bin = int(binned[bin_col].max())
    total_bins = max_bin + 1

    X = np.zeros((total_bins, 9), dtype=float)
    y = np.full((total_bins,), label, dtype=int)

    for bid in range(total_bins):
        print(f"\rProcessing bin {bid + 1}/{total_bins}", end="", flush=True)
        grp = binned[binned[bin_col] == bid]

        count_type0 = int((grp["fc_type"] == 0).sum())
        bytes_type0 = float(grp.loc[grp["fc_type"] == 0, "frame_len"].sum())

        count_type2 = int((grp["fc_type"] == 2).sum())
        bytes_type2 = float(grp.loc[grp["fc_type"] == 2, "frame_len"].sum())

        total_bytes = float(grp["frame_len"].sum())

        if not grp.empty:
            min_size = float(grp["frame_len"].min())
            max_size = float(grp["frame_len"].max())
            mean_size = float(grp["frame_len"].mean())
            var_size = float(grp["frame_len"].var(ddof=0))
            if np.isnan(var_size):
                var_size = 0.0
        else:
            min_size = max_size = mean_size = var_size = 0.0

        X[bid] = [
            count_type0,
            bytes_type0,
            count_type2,
            bytes_type2,
            total_bytes,
            min_size,
            max_size,
            mean_size,
            var_size,
        ]

    print()
    return (X, y) if return_y else X


def plot_bytes_per_bin(time: pd.Series,
                       bin_counts: pd.Series,
                       ax=None,
                       figsize: tuple = (10, 4),
                       color: str = "C0",
                       marker: str = "o",
                       s: int = 6,
                       alpha: float = 0.7,
                       bin_width: float | None = None,
                       title: str | None = None,
                       show: bool = True):
    """Plot packet count per bin vs time using two 1D Series.

    Args:
        time: 1D `pd.Series` of bin times (seconds, numeric) — typically `bin_start`.
        bin_counts: 1D `pd.Series` of packet counts per bin (integers).
        ax: optional matplotlib Axes to plot into. If None, a new figure is created.
        figsize: figure size when creating a new figure.
        color, marker, s, alpha: styling for the plot when `bin_width` is None.
        bin_width: if provided, draws bars of this width at each `time` position.
        title: optional plot title.
        show: whether to call `plt.show()` when creating a new figure.

    Returns:
        The matplotlib Axes with the plot.
    """
    if not isinstance(time, pd.Series) or not isinstance(bin_counts, pd.Series):
        raise TypeError("`time` and `bin_counts` must be pandas Series objects")

    if len(time) != len(bin_counts):
        raise ValueError("`time` and `bin_counts` must have the same length")

    x = pd.to_numeric(time, errors="coerce")
    y = pd.to_numeric(bin_counts, errors="coerce")
    mask = x.notna() & y.notna()

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    if bin_width is not None:
        ax.bar(x[mask], y[mask], width=bin_width, color=color, alpha=alpha, align='edge')
    else:
        ax.plot(x[mask], y[mask], marker=marker, color=color, linestyle='-', markeredgecolor=color)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("packets per bin")
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.4)

    if created_fig and show:
        plt.show()

    return ax

