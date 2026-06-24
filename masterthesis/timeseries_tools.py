import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft
from python_speech_features import mfcc
import pywt
from scipy.io import wavfile
import random

#from __future__ import annotations
import numpy as np
from typing import Optional, Literal, Tuple, Dict

#### Heavy Noise Augmentation


from typing import Tuple, Optional, Dict, Literal
import numpy as np
import pandas as pd

# ------------------------
# Helpers
# ------------------------

def _ensure_2d(y: np.ndarray) -> Tuple[np.ndarray, bool]:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y[:, None], True
    if y.ndim == 2:
        return y, False
    raise ValueError("y must be shape (T,) or (T, C)")

def _alias_then_restore(
    y: np.ndarray,
    factor: int,
    method: Literal["zoh", "linear"] = "zoh",
) -> np.ndarray:
    """
    Simulate aliasing by naively downsampling (no anti-alias filter), then
    upsample back to the original length via zero-order hold ('zoh') or linear interpolation.
    y: (T, C), float32
    """
    T, C = y.shape
    if factor <= 1:
        return y.copy()

    # Downsample without anti-alias -> aliasing occurs at this step
    y_ds = y[::factor, :]  # (ceil(T/factor), C)

    if method == "zoh":
        # zero-order hold back to T
        y_up = np.repeat(y_ds, repeats=factor, axis=0)
        if y_up.shape[0] < T:
            # pad last sample if needed
            pad = np.repeat(y_ds[-1:, :], repeats=T - y_up.shape[0], axis=0)
            y_up = np.concatenate([y_up, pad], axis=0)
        return y_up[:T, :].astype(np.float32, copy=False)

    elif method == "linear":
        # per-channel linear interpolation back to original grid
        t_low = np.arange(y_ds.shape[0], dtype=np.float32) * factor
        t_full = np.arange(T, dtype=np.float32)
        y_up = np.empty_like(y)
        for c in range(C):
            y_up[:, c] = np.interp(t_full, t_low, y_ds[:, c]).astype(np.float32, copy=False)
        return y_up
    else:
        raise ValueError("method must be 'zoh' or 'linear'")

# ------------------------
# Um et al.-style time warping
# ------------------------

def _time_warp_um(
    y: np.ndarray,
    strength: float = 0.10,
    knots: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Smooth random time-warping similar in spirit to Um et al. (2017):
      1) sample small random offsets at a few 'knots'
      2) linearly interpolate to full length -> smooth curve
      3) exp(curve) -> positive 'speed' weights
      4) cumulative sum & normalize to [0, T-1] -> strictly increasing time map
      5) resample original signal at warped times (per channel)

    y: (T, C)
    strength: std of random offsets at knots (typ. 0.05..0.20)
    knots: number of control points for the warping curve (4–8 recommended)
    """
    if rng is None:
        rng = np.random.default_rng()
    y2d, squeezed = _ensure_2d(y)
    T, C = y2d.shape

    if T < 3 or strength <= 0.0 or knots < 2:
        return y.copy()

    # 1) random offsets at knots
    xk = np.linspace(0, T - 1, knots, dtype=np.float32)
    yk = rng.normal(0.0, strength, size=knots).astype(np.float32)

    # 2) smooth curve via linear interpolation (keeps deps minimal)
    curve = np.interp(np.arange(T, dtype=np.float32), xk, yk).astype(np.float32)

    # 3) positive speeds
    speed = np.exp(curve).astype(np.float32)

    # 4) strictly increasing time map in [0, T-1]
    tau = np.cumsum(speed)
    tau = (tau - tau[0]) / (tau[-1] - tau[0]) * (T - 1)

    # 5) resample per channel at warped times
    t_full = np.arange(T, dtype=np.float32)
    y_warp = np.empty_like(y2d)
    for c in range(C):
        y_warp[:, c] = np.interp(t_full, tau, y2d[:, c]).astype(np.float32, copy=False)

    return y_warp.squeeze(-1) if squeezed else y_warp

# ------------------------
# Main degradation chain (with optional time warping)
# ------------------------

def degrade_signal_chain(
    y: np.ndarray,
    sample_rate_hz: float,
    downsample_factor: int,
    resample_method: Literal["zoh", "linear"] = "zoh",
    n_spikes: int = 6,
    spike_scale_sigma: float = 8.0,
    clip_sigma: float = 2.5,
    noise_kwargs: Optional[Dict] = None,
    rng: Optional[np.random.Generator] = None,
    # NEW:
    time_warp_enabled: bool = False,
    time_warp_strength: float = 0.10,
    time_warp_knots: int = 4,
) -> np.ndarray:
    """
    Pipeline:
      (1) optional smooth time warping (Um-style)
      (2) aliasing via naive downsampling
      (3) random spikes
      (4) hard clipping
      (5) existing noise injection

    Returns array with same shape as y.
    """
    if rng is None:
        rng = np.random.default_rng()
    y2d, squeezed = _ensure_2d(y)
    T, C = y2d.shape

    # --- (1) Optional Time Warping (applied first so later artifacts follow warped time) ---
    if time_warp_enabled and time_warp_strength > 0.0:
        y_tw = _time_warp_um(y2d, strength=time_warp_strength, knots=time_warp_knots, rng=rng)
        y2d = y_tw if y_tw.ndim == 2 else y_tw[:, None]

    # --- (2) Aliasing ---
    y_alias = _alias_then_restore(y2d, factor=downsample_factor, method=resample_method)

    # --- (3) Add spikes (per-channel) ---
    y_aug = y_alias.copy()
    n_spikes = max(0, int(n_spikes))
    if n_spikes > 0:
        spike_idx = rng.choice(T, size=min(n_spikes, T), replace=False)
        for c in range(C):
            sd = float(y_aug[:, c].std() or 1.0)
            amp = spike_scale_sigma * sd
            signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=len(spike_idx))
            y_aug[spike_idx, c] = y_aug[spike_idx, c] + signs * amp

    # --- (4) Hard clipping (per-channel) ---
    for c in range(C):
        sd = float(y_aug[:, c].std() or 1.0)
        limit = clip_sigma * sd
        y_aug[:, c] = np.clip(y_aug[:, c], -limit, limit)

    # --- (5) Existing noise augmentation ---
    nk = dict(dtype=np.float32)
    if noise_kwargs:
        nk.update(noise_kwargs)
    nk.setdefault("sample_rate_hz", sample_rate_hz)  # e.g., if periodic component present
    y_noisy = noisy_augment_signal(y_aug, **nk)

    # shape back
    if squeezed:
        return np.asarray(y_noisy, dtype=np.float32).squeeze(-1)
    return np.asarray(y_noisy, dtype=np.float32)
# ------------------------
# Six levels 
# ------------------------

def apply_six_levels_and_save(
    csv_path: str,
    out_prefix: str,
    cols: Tuple[str, ...] = ("acc_y_combined",),
    sample_rate_hz: float = 100.0,
) -> None:
    """
    Loads a CSV, applies six levels of degradation (only time-warping differs),
    and saves six CSVs.
    """
    df = pd.read_csv(csv_path)
    y = df[list(cols)].to_numpy(np.float32)  # (T, C)
    rng = np.random.default_rng(42)

    LEVELS = {
    "_degraded": dict(
        downsample_factor=2,  # was 4 → weaker resampling
        resample_method="zoh",
        n_spikes=5,  # was 10 → half the number of spikes
        spike_scale_sigma=6.0,  # was 12.0 → weaker spike amplitude
        clip_sigma=4.0,  # was 2.0 → less aggressive clipping (higher sigma = fewer clipped samples)
        noise_kwargs=dict(
            noise_color="mixture",
            mixture_weights=(0.3, 0.4, 0.3),
            target_snr_db=2.0,  # was -1.0 → slightly higher SNR (less noise)
            periodic_amp=0.5,  # was 1.0 → weaker periodic noise
            periodic_freq_hz=6.0,
            sample_rate_hz=sample_rate_hz,
        ),
        time_warp_enabled=True,
        time_warp_strength=0.375,  # was 0.75 → 50% weaker time distortion
        time_warp_knots=100,
    ),
}



    for name, params in LEVELS.items():
        y_deg = degrade_signal_chain(
            y=y,
            sample_rate_hz=sample_rate_hz,
            rng=rng,
            **params,
        )
        df_out = df.copy()
        y_deg = np.asarray(y_deg, dtype=np.float32).reshape(-1, len(cols))
        df_out[list(cols)] = y_deg
        df_out.to_csv(
            f"/home/ws/ugoby/master_thesis/data_conference/labels/degraded/{out_prefix}_{name}.csv",
            index=False,
        )






#### Old Aug below

def ensemble_augment_signal(
    y: np.ndarray,
    std_min: float = 0.1,
    std_max: float = 0.2,
    n_ensembles: int = 100,
    rng: Optional[np.random.Generator] = None,
    dtype=np.float32,
) -> np.ndarray:
    """
    Noise-assisted *ensemble augmentation* for 1-D time series (optionally multi-channel).

    Closely follows the paper's idea:
      1) Draw multiple (j) white-noise realizations with SD sampled uniformly in [std_min, std_max]
      2) Add to the original in positive and negative pairs
      3) Mean the positive set and the negative set separately
      4) Average those two means to get the augmented signal z

    This keeps the signal near the original while injecting subtle, realistic variability.
    Defaults reflect the paper's validated settings (Stdmin≈0.1, Stdmax≈0.2; j≈100).  # See paper.
    
    Args
    ----
    y : np.ndarray
        Input time series. Shape (T,) for single-channel or (T, C) for multi-channel.
    std_min, std_max : float
        Lower/upper bounds for per-ensemble noise standard deviation.
    n_ensembles : int
        Number of ensembles (j). Around 100 was found sufficient in the paper.
    rng : np.random.Generator, optional
        Random generator. If None, uses np.random.default_rng().
    dtype : np.dtype
        Floating dtype of outputs.

    Returns
    -------
    z : np.ndarray
        Augmented signal with same shape as y and dtype=dtype.

    Notes
    -----
    - If you want *more* deviation from the original, raise std_max (and/or std_min).
    - If y is integer-typed, it will be cast to float.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Validate input and coerce to (T, C)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]  # (T, 1)
    elif y.ndim != 2:
        raise ValueError("y must be shape (T,) or (T, C)")

    T, C = y.shape
    if not (0 <= std_min < std_max):
        raise ValueError("Require 0 <= std_min < std_max")
    if n_ensembles <= 0:
        raise ValueError("n_ensembles must be positive")

    # Expand to (j, T, C)
    y_exp = y[None, :, :]  # (1, T, C)
    # Per-ensemble SDs (broadcast across time)
    stds = rng.uniform(std_min, std_max, size=(n_ensembles, 1, C)).astype(np.float32)

    # Positive and negative noise ensembles
    noise_pos = rng.normal(0.0, 1.0, size=(n_ensembles, T, C)).astype(np.float32) * stds
    noise_neg = rng.normal(0.0, 1.0, size=(n_ensembles, T, C)).astype(np.float32) * stds

    # Means across ensembles
    y1 = (y_exp + noise_pos).mean(axis=0)   # (T, C)
    y2 = (y_exp - noise_neg).mean(axis=0)   # (T, C)

    z = 0.5 * (y1 + y2)                      # (T, C)

    # Squeeze back to (T,) if original was 1-D
    z = z.astype(dtype, copy=False)
    return z.squeeze(-1) if z.shape[1] == 1 else z


def noisy_augment_signal(
    y: np.ndarray,
    noise_color: Literal["white", "pink", "brown", "periodic", "mixture"] = "white",
    noise_std: Optional[float] = None,
    target_snr_db: Optional[float] = None,
    mixture_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),  # white, pink, brown
    periodic_amp: float = 0.0,
    periodic_freq_hz: float = 5.0,
    sample_rate_hz: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    dtype=np.float32,
) -> np.ndarray:
    """
    Deliberately *degrade* SNR by adding noise. Offers:
      - White noise
      - Pink (1/f) and Brown (1/f^2) colored noise (via frequency-domain shaping)
      - Periodic component (e.g., mechanical cyclic interference)
      - Exact SNR targeting per channel (if target_snr_db is given), else scale by noise_std·std(signal)

    Args
    ----
    y : np.ndarray
        Input time series. Shape (T,) or (T, C).
    noise_color : {'white','pink','brown','periodic','mixture'}
        Noise type. 'mixture' uses mixture_weights of white/pink/brown.
    noise_std : float, optional
        If set and target_snr_db is None, scales noise as noise_std * std(y) per channel.
    target_snr_db : float, optional
        If set, the output y_noisy will have approximately this SNR (per channel).
    mixture_weights : (w_white, w_pink, w_brown)
        Weights for 'mixture' noise; will be normalized internally.
    periodic_amp : float
        Amplitude of optional sinusoidal interference. Set >0 to include.
    periodic_freq_hz : float
        Frequency (Hz) for the periodic component (requires sample_rate_hz).
    sample_rate_hz : float, optional
        Needed only if periodic component is used.
    rng : np.random.Generator, optional
        Random generator. If None, uses np.random.default_rng().
    dtype : np.dtype
        Floating dtype of outputs.

    Returns
    -------
    y_noisy : np.ndarray
        y plus the constructed noise, same shape/type handling as ensemble_augment_signal.

    Notes
    -----
    - If both target_snr_db and noise_std are None, defaults to noise_std=0.3 (30% of per-channel std).
    - Colored noise is produced by shaping white-noise spectrum (simple 1/√f for pink, 1/f for brown).
    """
    if rng is None:
        rng = np.random.default_rng()

    y = np.asarray(y, dtype=np.float32)
    squeeze = False
    if y.ndim == 1:
        y = y[:, None]
        squeeze = True
    elif y.ndim != 2:
        raise ValueError("y must be shape (T,) or (T, C)")

    T, C = y.shape

    def _fft_colored(shape_filter: Literal["pink", "brown"], length: int, rng: np.random.Generator) -> np.ndarray:
        """
        Generate a single-channel colored-noise vector of length `length`
        by shaping white noise in the frequency domain.
        """
        # White noise
        x = rng.normal(0.0, 1.0, size=length).astype(np.float32)
        X = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(length, d=1.0)  # normalized (no absolute Hz dependency)
        freqs[0] = freqs[1] if len(freqs) > 1 else 1.0  # avoid division by zero

        if shape_filter == "pink":
            S = 1.0 / np.sqrt(freqs)          # ~1/√f
        elif shape_filter == "brown":
            S = 1.0 / (freqs)                 # ~1/f
        else:
            raise ValueError("shape_filter must be 'pink' or 'brown'.")

        Y = X * S
        y_col = np.fft.irfft(Y, n=length).astype(np.float32)
        # Normalize to unit std (approx) to allow easy scaling later
        sd = y_col.std() or 1.0
        return (y_col / sd).astype(np.float32)

    # Build noise per channel
    noise = np.zeros_like(y, dtype=np.float32)
    for c in range(C):
        if noise_color == "white":
            n = rng.normal(0.0, 1.0, size=T).astype(np.float32)
        elif noise_color in ("pink", "brown"):
            n = _fft_colored("pink" if noise_color == "pink" else "brown", T, rng)
        elif noise_color == "periodic":
            if sample_rate_hz is None:
                raise ValueError("sample_rate_hz is required when noise_color='periodic'.")
            t = np.arange(T, dtype=np.float32) / float(sample_rate_hz)
            n = np.sin(2.0 * np.pi * periodic_freq_hz * t).astype(np.float32)
            # Normalize to unit std
            n = n / (n.std() or 1.0)
        elif noise_color == "mixture":
            w_w, w_p, w_b = mixture_weights
            wsum = (w_w + w_p + w_b) or 1.0
            w_w, w_p, w_b = w_w / wsum, w_p / wsum, w_b / wsum
            n_w = rng.normal(0.0, 1.0, size=T).astype(np.float32)
            n_p = _fft_colored("pink", T, rng)
            n_b = _fft_colored("brown", T, rng)
            n = (w_w * n_w + w_p * n_p + w_b * n_b).astype(np.float32)
            # Normalize to unit std
            n = n / (n.std() or 1.0)
        else:
            raise ValueError("Unsupported noise_color.")

        # Optional periodic *addition* on top of stochastic noise
        if periodic_amp > 0.0:
            if sample_rate_hz is None:
                raise ValueError("sample_rate_hz is required when adding periodic component.")
            t = np.arange(T, dtype=np.float32) / float(sample_rate_hz)
            periodic = np.sin(2.0 * np.pi * periodic_freq_hz * t).astype(np.float32)
            periodic = periodic / (periodic.std() or 1.0)
            n = n + periodic_amp * periodic

        # Scale to hit target SNR or noise_std*std(signal)
        sig = y[:, c]
        sig_power = float(np.mean(sig.astype(np.float64) ** 2))
        if target_snr_db is not None:
            # p_noise = p_signal / 10^(SNR/10)
            p_noise = sig_power / (10.0 ** (target_snr_db / 10.0))
            # current noise is ~unit std => unit power ~1
            scale = np.sqrt(max(p_noise, 1e-12))
        else:
            if noise_std is None:
                noise_std = 0.3  # default 30% of per-channel std
            scale = float(noise_std) * (sig.std() or 1.0)

        noise[:, c] = (n * scale).astype(np.float32)

    y_noisy = (y + noise).astype(dtype, copy=False)
    return y_noisy.squeeze(-1) if squeeze else y_noisy


# -------------------------
# Example usage (commented)
# -------------------------
# import pandas as pd
# df = pd.read_csv("your_accelerometer.csv")
# cols = ["acc_x_dashboard", "acc_y_dashboard", "acc_z_dashboard"]
# y = df[cols].to_numpy()  # shape (T, 3)
#
# # 1) Faithful ensemble augmentation (paper-like):
# y_aug = ensemble_augment_signal(y, std_min=0.1, std_max=0.2, n_ensembles=100)
#
# # 2) Deliberately degrade SNR to 5 dB using pink noise:
# y_noisy = noisy_augment_signal(y, noise_color="pink", target_snr_db=5.0)
#
# df_aug = df.copy()
# df_aug[cols] = y_aug
# df_noisy = df.copy()
# df_noisy[cols] = y_noisy


#this function generates MFCC and CWT images from a DataFrame containing time series data 
#the images amount to exactly 33090 / 15717


def generate_images_fixed_count(
    df, data_column, method, output_folder,
    target_count, window_size=50, target_filenames=None
):
    """
    Generate exactly `target_count` spectrograms with sliding windows.
    Save each spectrogram using `target_filenames[i]` (must have len==target_count).
    Dominant label is computed from the window; deviations vs filename label are counted.
    """
    import os, numpy as np, matplotlib.pyplot as plt, pywt
    from python_speech_features import mfcc
    from tqdm import tqdm

    os.makedirs(output_folder, exist_ok=True)

    if 'image_filename' not in df.columns:
        raise ValueError("CSV must contain 'image_filename' column.")
    label_cols = ['dirt_road', 'cobblestone_road', 'asphalt_road']
    for c in label_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in DataFrame.")

    if target_filenames is None or len(target_filenames) != target_count:
        raise ValueError("Provide `target_filenames` with length == target_count.")

    total_rows = len(df)
    if total_rows < window_size:
        raise ValueError("CSV shorter than window size.")

    # Evenly spaced start indices to hit exactly target_count windows
    if target_count == 1:
        starts = [0]
    else:
        span = total_rows - window_size
        starts = [int(round(i * span / (target_count - 1))) for i in range(target_count)]

    results, deviation_count = [], 0
    pbar = tqdm(total=target_count, desc="Generating MFCC spectrograms", unit="img")

    for i, start in enumerate(starts):
        end = start + window_size
        window = df.iloc[start:end]
        signal = window[data_column].to_numpy()
        if signal.size < 2:
            pbar.update(1); continue

        # dominant road type in window
        sums = window[label_cols].sum()
        dom_label = sums.idxmax().replace('_road', '')

        # filename to save (unique, from the video folder listing)
        fname = target_filenames[i]
        # compare to filename label (suffix)
        try:
            frame_label = os.path.splitext(fname)[0].split('_', 1)[1]
            if frame_label != dom_label:
                deviation_count += 1
        except Exception:
            pass

        save_path = os.path.join(output_folder, fname)
        plt.figure(figsize=(5, 5), dpi=100)
        if method.lower() == 'mfcc':
            feat = mfcc(signal, samplerate=100, numcep=40, nfilt=40)
            plt.imshow(feat.T, aspect='auto', origin='lower', cmap='viridis')
        elif method.lower() == 'cwt':
            coefs, _ = pywt.cwt(signal, np.arange(1, 31), 'morl')
            plt.imshow(np.abs(coefs), aspect='auto', origin='lower', cmap='viridis')
        else:
            raise ValueError("method must be 'mfcc' or 'cwt'")
        plt.axis('off'); plt.tight_layout(pad=0)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0); plt.close()

        results.append({
            'image_filename': fname,
            'dominant_label': dom_label,
            'start_index': int(window.index.min()),
            'end_index': int(window.index.max()),
        })
        pbar.update(1)

    pbar.close()
    print("\n=== Spectrogram Generation Summary ===")
    print(f"Spectrograms generated: {len(results)} (target {target_count})")
    print(f"Output folder: {output_folder}")
    print(f"Label deviations (window dominant vs filename suffix): {deviation_count}")
    print("✅ Exact count & 1:1 filenames with frames.")
    return pd.DataFrame(results)


def synchronize_image_folders(folder_variable_map: dict, image_format=".jpg"):
    """
    Synchronize and rename image files across multiple folders to shared names like 00001_asphalt.jpg.

    Parameters:
        folder_variable_map (dict): Ordered mapping {folder_path: variable_name}.
                                    The first folder is used as the label source.
        image_format (str): File extension (e.g., '.jpg')

    Returns:
        int: Number of images renamed
    """
    folder_files = {}
    folder_paths = list(folder_variable_map.keys())

    for folder in folder_paths:
        files = sorted([f for f in os.listdir(folder) if f.endswith(image_format)])
        folder_files[folder] = files

    lengths = [len(flist) for flist in folder_files.values()]
    if not all(l == lengths[0] for l in lengths):
        raise ValueError(f"Folders contain differing image counts: {lengths}")

    count = lengths[0]
    label_source_folder = folder_paths[0]

    for i in range(1, count + 1):
        base_index = f"{i:05d}"

        # Extract label from the label source folder
        original_label_file = folder_files[label_source_folder][i - 1]
        base_part = os.path.splitext(original_label_file)[0]
        label = base_part.split("_")[-1]

        new_shared_name = f"{base_index}_{label}{image_format}"

        for folder in folder_paths:
            old_name = folder_files[folder][i - 1]
            new_path = os.path.join(folder, new_shared_name)
            os.rename(os.path.join(folder, old_name), new_path)

    return count


def generate_mfcc_images(df, data_column, output_folder='mfcc_images'):
    """
    Generate MFCC images from time series data using sliding windows.
    Overlapping samples are labeled as 'mixed_<majority>' (e.g. mixed_asphalt),
    so they can later be filtered or grouped flexibly.
    """
    # ===== SETTINGS SECTION =====
    WINDOW_SIZE = 50           # Number of data points per image
    OVERLAP_RATIO = 0.80       # 80% overlap between windows
    MIN_CLASS_RATIO = 0.8      # Minimum 80% of window must be same class
    IMAGE_SIZE = (500, 500)
    IMAGE_FORMAT = '.jpg'
    MFCC_COEFFS = 40
    SAMPLING_RATE = 100
    # ============================

    os.makedirs(output_folder, exist_ok=True)

    signal_data = df[data_column].values
    label_cols = ['dirt_road', 'cobblestone_road', 'asphalt_road']
    step_size = int(WINDOW_SIZE * (1 - OVERLAP_RATIO))

    results = []
    image_counter = 1

    for start_idx in range(0, len(signal_data) - WINDOW_SIZE + 1, step_size):
        end_idx = start_idx + WINDOW_SIZE
        window_signal = signal_data[start_idx:end_idx]
        window_labels = df[label_cols].iloc[start_idx:end_idx]

        # Determine label proportions
        label_sums = window_labels.sum()
        total_samples = len(window_labels)

        ratios = {col: label_sums[col] / total_samples for col in label_cols}
        majority_col = max(ratios, key=ratios.get)
        majority_ratio = ratios[majority_col]
        majority_class = majority_col.replace('_road', '')

        # Labeling rule
        if majority_ratio >= MIN_CLASS_RATIO:
            label = majority_class
        else:
            label = f"mixed_{majority_class}"

        # Generate MFCC
        mfcc_features = mfcc(window_signal, samplerate=SAMPLING_RATE,
                             numcep=MFCC_COEFFS, nfilt=MFCC_COEFFS)

        # Save image
        image_filename = f"{image_counter:05d}_{label}{IMAGE_FORMAT}"
        image_path = os.path.join(output_folder, image_filename)
        plt.figure(figsize=(IMAGE_SIZE[0]/100, IMAGE_SIZE[1]/100), dpi=100)
        plt.imshow(mfcc_features.T, aspect='auto', origin='lower', cmap='viridis')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Store metadata
        results.append({
            'image_filename': image_filename,
            'label': label,
            'majority_class': majority_class,
            'majority_ratio': majority_ratio,
            'start_timestamp': df['timestamp'].iloc[start_idx],
            'end_timestamp': df['timestamp'].iloc[end_idx - 1],
            'window_start_idx': start_idx,
            'window_end_idx': end_idx - 1
        })

        image_counter += 1

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_folder, 'mfcc_results.csv'), index=False)
    return results_df


########


def generate_cwt_images(df, data_column, output_folder='cwt_images'):
    """
    Generate CWT (Continuous Wavelet Transform) images from time series data using sliding windows.
    
    Parameters:
    df (pandas.DataFrame): Input dataset with sensor data and labels
    data_column (str): Column name to process (e.g., 'acc_y_combined')
    output_folder (str): Name of output folder for images
    
    Returns:
    pandas.DataFrame: New CSV with image filenames and corresponding labels
    """
    
    # ===== SETTINGS SECTION =====
    WINDOW_SIZE = 50           # Number of data points per image
    OVERLAP_RATIO = 0.5        # 50% overlap between windows
    MIN_CLASS_RATIO = 0.8      # Minimum 80% of window must be same class
    IMAGE_SIZE = (500, 500)    # Image dimensions
    IMAGE_FORMAT = '.jpg'      # Output format
    CWT_WAVELET = 'morl'       # Options: 'morl', 'mexh', 'cgau8', 'shan'
    CWT_SCALES = np.arange(1, 31)  # Frequency scales for CWT
    SAMPLING_RATE = 100        # Assumed sampling rate (Hz)
    # ============================
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract signal data
    signal_data = df[data_column].values
    
    # Label columns
    label_cols = ['dirt_road', 'cobblestone_road', 'asphalt_road']
    
    # Calculate step size for sliding window
    step_size = int(WINDOW_SIZE * (1 - OVERLAP_RATIO))
    
    # Results storage
    results = []
    image_counter = 1
    
    # Generate sliding windows
    for start_idx in range(0, len(signal_data) - WINDOW_SIZE + 1, step_size):
        end_idx = start_idx + WINDOW_SIZE
        
        # Extract window data
        window_signal = signal_data[start_idx:end_idx]
        window_labels = df[label_cols].iloc[start_idx:end_idx]
        
        # Determine dominant label
        label_sums = window_labels.sum()
        total_samples = len(window_labels)
        
        # Find dominant class
        dominant_class = None
        max_ratio = 0
        
        for col in label_cols:
            ratio = label_sums[col] / total_samples
            if ratio > max_ratio:
                max_ratio = ratio
                if ratio >= MIN_CLASS_RATIO:
                    dominant_class = col.replace('_road', '')
        
        # Assign label
        if dominant_class is None:
            label = 'mixed'
        else:
            label = dominant_class
        
        # Generate CWT
        coefficients, frequencies = pywt.cwt(window_signal, CWT_SCALES, CWT_WAVELET)
        
        # Create image filename
        image_filename = f"{image_counter:03d}_{label}{IMAGE_FORMAT}"
        image_path = os.path.join(output_folder, image_filename)
        
        # Generate and save CWT image
        plt.figure(figsize=(IMAGE_SIZE[0]/100, IMAGE_SIZE[1]/100), dpi=100)
        plt.imshow(np.abs(coefficients), aspect='auto', origin='lower', cmap='viridis')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Store result
        results.append({
            'image_filename': image_filename,
            'label': label,
            'start_timestamp': df['timestamp'].iloc[start_idx],
            'end_timestamp': df['timestamp'].iloc[end_idx-1],
            'window_start_idx': start_idx,
            'window_end_idx': end_idx-1
        })
        
        image_counter += 1
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_folder, 'cwt_results.csv'), index=False)
    return results_df


def combine_sensor_data(csv_file_path):
    """
    Add combined accelerometer and gyroscope Y-axis data to the original DataFrame.
    
    Parameters:
    csv_file_path (str): Path to the CSV file containing sensor data
    
    Returns:
    pandas.DataFrame: Original DataFrame with added acc_y_combined and gyro_y_combined columns
    """
    
    df = pd.read_csv(csv_file_path)
    
    df['acc_y_combined'] = (df['acc_y_dashboard'] + 
                            df['acc_y_above_suspension'] + 
                            df['acc_y_below_suspension'])

    df['gyro_y_combined'] = (df['gyro_y_dashboard'] + 
                             df['gyro_y_above_suspension'] + 
                             df['gyro_y_below_suspension'])
    
    df['temp_combined'] = ((df['temp_dashboard'] + 
                             df['temp_above_suspension'] + 
                             df['temp_below_suspension']) / 3)

    return df


def plot_acc_y_timeseries(csv_file_path, figsize=(12, 8), save_path=None):
    """
    Plot acceleration Y data from three different locations on a timeseries graph.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the CSV file containing the sensor data
    figsize : tuple, optional
        Figure size as (width, height). Default is (12, 8)
    save_path : str, optional
        Path to save the plot. If None, plot is displayed but not saved
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Convert timestamp to datetime for better x-axis formatting
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the three acc_y measurements
    ax.plot(df['datetime'], df['acc_y_dashboard'], 
            label='Dashboard', linewidth=2, alpha=0.8)
    ax.plot(df['datetime'], df['acc_y_above_suspension'], 
            label='Above Suspension', linewidth=2, alpha=0.8)
    ax.plot(df['datetime'], df['acc_y_below_suspension'], 
            label='Below Suspension', linewidth=2, alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Acceleration Y (m/s²)', fontsize=12)
    ax.set_title('Y-Axis Acceleration Comparison Across Sensor Locations', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis to show time nicely
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
 
    plt.show()
    
    return fig