## old mfcc generation

def generate_images_fixed_count(df, data_column, method, output_folder, target_count=33090, window_size=50):
    """
    Generate exactly `target_count` images using MFCC or CWT from time series data.
    Automatically adjusts overlap ratio. Includes progress bar.

    Parameters:
        df (pd.DataFrame): Input data
        data_column (str): Column name to use for transformation
        method (str): 'mfcc' or 'cwt'
        output_folder (str): Destination folder
        target_count (int): Number of images to generate
        window_size (int): Sliding window size

    Returns:
        pd.DataFrame: DataFrame with image metadata
    """
    os.makedirs(output_folder, exist_ok=True)
    signal_data = df[data_column].values
    total_rows = len(signal_data)

    # Compute needed overlap ratio
    step_size = (total_rows - window_size) / (target_count - 1)
    overlap_ratio = 1 - (step_size / window_size)
    overlap_ratio = max(0.0, min(overlap_ratio, 0.99))  # keep safe bounds

    step_size = int(window_size * (1 - overlap_ratio))
    label_cols = ['dirt_road', 'cobblestone_road', 'asphalt_road']
    image_size = (500, 500)
    image_format = '.jpg'

    results = []
    image_counter = 1

    pbar = tqdm(total=target_count, desc=f"Generating {method.upper()} images", unit="img")

    for start_idx in range(0, len(signal_data) - window_size + 1, step_size):
        if image_counter > target_count:
            break

        end_idx = start_idx + window_size
        window_signal = signal_data[start_idx:end_idx]
        window_labels = df[label_cols].iloc[start_idx:end_idx]
        label_sums = window_labels.sum()
        label = 'mixed'
        for col in label_cols:
            if label_sums[col] / window_size >= 0.8:
                label = col.replace('_road', '')
                break

        filename = f"{image_counter:05d}_{label}{image_format}"
        path = os.path.join(output_folder, filename)

        # Generate MFCC
        if method == 'mfcc':
            mfcc_features = mfcc(window_signal, samplerate=100, numcep=40, nfilt=40)
            plt.figure(figsize=(image_size[0]/100, image_size[1]/100), dpi=100)
            plt.imshow(mfcc_features.T, aspect='auto', origin='lower', cmap='viridis')
        # Generate CWT
        elif method == 'cwt':
            coefs, _ = pywt.cwt(window_signal, np.arange(1, 31), 'morl')
            plt.figure(figsize=(image_size[0]/100, image_size[1]/100), dpi=100)
            plt.imshow(np.abs(coefs), aspect='auto', origin='lower', cmap='viridis')
        else:
            raise ValueError("method must be 'mfcc' or 'cwt'")

        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()

        results.append({
            'image_filename': filename,
            'label': label,
            'start_index': start_idx,
            'end_index': end_idx - 1
        })

        image_counter += 1
        pbar.update(1)

    pbar.close()
    return pd.DataFrame(results)







## Time Series augmentation

# assumes your noisy_augment_signal(y, ...) is already imported as in your snippet
# from timeseries_tools import noisy_augment_signal

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
) -> np.ndarray:
    """
    Pipeline: (2) aliasing via naive downsampling -> (4) random spikes -> hard clipping -> existing noise.
    Returns array with same shape as y.
    """
    if rng is None:
        rng = np.random.default_rng()
    y2d, squeezed = _ensure_2d(y)
    T, C = y2d.shape

    # --- (2) Aliasing ---
    y_alias = _alias_then_restore(y2d, factor=downsample_factor, method=resample_method)

    # --- (4a) Add spikes (per-channel) ---
    y_aug = y_alias.copy()
    n_spikes = max(0, int(n_spikes))
    if n_spikes > 0:
        spike_idx = rng.choice(T, size=min(n_spikes, T), replace=False)
        for c in range(C):
            sd = float(y_aug[:, c].std() or 1.0)
            amp = spike_scale_sigma * sd
            signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=len(spike_idx))
            y_aug[spike_idx, c] = y_aug[spike_idx, c] + signs * amp

    # --- (4b) Hard clipping (per-channel) ---
    for c in range(C):
        sd = float(y_aug[:, c].std() or 1.0)
        limit = clip_sigma * sd
        y_aug[:, c] = np.clip(y_aug[:, c], -limit, limit)

    # --- Add your existing noise augmentation ---
    nk = dict(dtype=np.float32)
    if noise_kwargs:
        nk.update(noise_kwargs)
    nk.setdefault("sample_rate_hz", sample_rate_hz)  # needed if periodic component present
    y_noisy = noisy_augment_signal(y_aug, **nk)

    # shape back
    if squeezed:
        return np.asarray(y_noisy, dtype=np.float32).squeeze(-1)
    return np.asarray(y_noisy, dtype=np.float32)


# ------------------------
# Four stronger levels (L1..L4)
# ------------------------

def apply_four_levels_and_save(
    csv_path: str,
    out_prefix: str,
    cols: Tuple[str, ...] = ("acc_y_combined",),
    sample_rate_hz: float = 100.0,
) -> None:
    """
    Loads a CSV, applies four increasingly strong degradations, and saves four CSVs.
    """
    df = pd.read_csv(csv_path)
    y = df[list(cols)].to_numpy(np.float32)  # (T, C)

    rng = np.random.default_rng(42)

    LEVELS = {
    "1_pvs9_mfcc++": dict(  # -10.0 dB
        downsample_factor=4, resample_method="zoh",
        n_spikes=10, spike_scale_sigma=12.0, clip_sigma=2.0,
        noise_kwargs=dict(
            noise_color="mixture", mixture_weights=(0.3, 0.4, 0.3),
            target_snr_db=-25,
            periodic_amp=1.0, periodic_freq_hz=6.0, sample_rate_hz=sample_rate_hz
        )
    ),
    "2_pvs9_mfcc++": dict(  # -10.0 dB
        downsample_factor=4, resample_method="zoh",
        n_spikes=10, spike_scale_sigma=12.0, clip_sigma=2.0,
        noise_kwargs=dict(
            noise_color="mixture", mixture_weights=(0.3, 0.4, 0.3),
            target_snr_db=-25,
            periodic_amp=1.0, periodic_freq_hz=6.0, sample_rate_hz=sample_rate_hz
        )
    ),
    "3_pvs9_mfcc++": dict(  # -10.0 dB
        downsample_factor=4, resample_method="zoh",
        n_spikes=10, spike_scale_sigma=12.0, clip_sigma=2.0,
        noise_kwargs=dict(
            noise_color="mixture", mixture_weights=(0.3, 0.4, 0.3),
            target_snr_db=-25,
            periodic_amp=1.0, periodic_freq_hz=6.0, sample_rate_hz=sample_rate_hz
        )
    ),
    "4_pvs9_mfcc++": dict(  # -10.0 dB
        downsample_factor=4, resample_method="zoh",
        n_spikes=10, spike_scale_sigma=12.0, clip_sigma=2.0,
        noise_kwargs=dict(
            noise_color="mixture", mixture_weights=(0.3, 0.4, 0.3),
            target_snr_db=-25,
            periodic_amp=1.0, periodic_freq_hz=6.0, sample_rate_hz=sample_rate_hz
        )
    ),
    "5_pvs9_mfcc++": dict(  # -10.0 dB
        downsample_factor=4, resample_method="zoh",
        n_spikes=10, spike_scale_sigma=12.0, clip_sigma=2.0,
        noise_kwargs=dict(
            noise_color="mixture", mixture_weights=(0.3, 0.4, 0.3),
            target_snr_db=-25,
            periodic_amp=1.0, periodic_freq_hz=6.0, sample_rate_hz=sample_rate_hz
        )
    ),
    "6_pvs9_mfcc++": dict(  # -10.0 dB
        downsample_factor=4, resample_method="zoh",
        n_spikes=10, spike_scale_sigma=12.0, clip_sigma=2.0,
        noise_kwargs=dict(
            noise_color="mixture", mixture_weights=(0.3, 0.4, 0.3),
            target_snr_db=-25,
            periodic_amp=1.0, periodic_freq_hz=6.0, sample_rate_hz=sample_rate_hz
        )
    ),
}






    for name, params in LEVELS.items():
        y_deg = degrade_signal_chain(
            y=y,
            sample_rate_hz=sample_rate_hz,
            rng=rng,
            **params
        )
        df_out = df.copy()
        # Ensure shape (T, len(cols))
        y_deg = np.asarray(y_deg, dtype=np.float32).reshape(-1, len(cols))
        df_out[list(cols)] = y_deg


        df_out.to_csv(f"/home/ws/ugoby/master_thesis/data/labels/{out_prefix}_{name}.csv", index=False)

# ------------------------
# Example usage:
# apply_four_levels_and_save(
#     csv_path="/home/ws/ugoby/master_thesis/data/pvs1_X01_label.csv",
#     out_prefix="df_degraded",
#     cols=("acc_y_combined",),
#     sample_rate_hz=100.0
# )



'''




"1_pvs9_mfcc+": dict( 
            downsample_factor=4, resample_method="zoh",
            n_spikes=10, spike_scale_sigma=12.0, clip_sigma=2.0,
            noise_kwargs=dict(
                noise_color="mixture", mixture_weights=(0.3, 0.4, 0.3),
                target_snr_db=-9.5,
                periodic_amp=1.0, periodic_freq_hz=6.0, sample_rate_hz=sample_rate_hz
            ),
            time_warp_enabled=False,
            time_warp_strength=0,
            time_warp_knots=0,
        ),
        
        "2_pvs9_mfcc++": dict(
            downsample_factor=4, resample_method="zoh",
            n_spikes=10, spike_scale_sigma=12.0, clip_sigma=2.0,
            noise_kwargs=dict(
                noise_color="mixture", mixture_weights=(0.3, 0.4, 0.3),
                target_snr_db=-10.0,
                periodic_amp=1.0, periodic_freq_hz=6.0, sample_rate_hz=sample_rate_hz,
            ),
            time_warp_enabled=True,
            time_warp_strength=7,
            time_warp_knots=4,
        ),
        "3_pvs9_mfcc++": dict(
            downsample_factor=4, resample_method="zoh",
            n_spikes=10, spike_scale_sigma=12.0, clip_sigma=2.0,
            noise_kwargs=dict(
                noise_color="mixture", mixture_weights=(0.3, 0.4, 0.3),
                target_snr_db=-10.0,
                periodic_amp=1.0, periodic_freq_hz=6.0, sample_rate_hz=sample_rate_hz,
            ),
            time_warp_enabled=True,
            time_warp_strength=10,
            time_warp_knots=30,
        ),
        "4_pvs9_mfcc++": dict(
            downsample_factor=4, resample_method="zoh",
            n_spikes=10, spike_scale_sigma=12.0, clip_sigma=2.0,
            noise_kwargs=dict(
                noise_color="mixture", mixture_weights=(0.3, 0.4, 0.3),
                target_snr_db=-10.0,
                periodic_amp=1.0, periodic_freq_hz=6.0, sample_rate_hz=sample_rate_hz,
            ),
            time_warp_enabled=True,
            time_warp_strength=10,
            time_warp_knots=60,
        ),
        "5_pvs9_mfcc++": dict(
            downsample_factor=4, resample_method="zoh",
            n_spikes=10, spike_scale_sigma=12.0, clip_sigma=2.0,
            noise_kwargs=dict(
                noise_color="mixture", mixture_weights=(0.3, 0.4, 0.3),
                target_snr_db=-10.0,
                periodic_amp=1.0, periodic_freq_hz=6.0, sample_rate_hz=sample_rate_hz,
            ),
            time_warp_enabled=True,
            time_warp_strength=10,
            time_warp_knots=120,
        ),
        "6_pvs9_mfcc++": dict(
            downsample_factor=4, resample_method="zoh",
            n_spikes=10, spike_scale_sigma=12.0, clip_sigma=2.0,
            noise_kwargs=dict(
                noise_color="mixture", mixture_weights=(0.3, 0.4, 0.3),
                target_snr_db=-10.0,
                periodic_amp=1.0, periodic_freq_hz=6.0, sample_rate_hz=sample_rate_hz,
            ),
            time_warp_enabled=True,
            time_warp_strength=10,
            time_warp_knots=200,
        ),


        

        ###





def generate_mfcc_images():

    timeseries_tools.apply_six_levels_and_save(
        csv_path="/home/ws/ugoby/master_thesis/data/pvs9_X01_label.csv",
        out_prefix="degraded",          # will create df_degraded_L1.csv ... L4.csv
        cols=("acc_y_combined",),          # which columns to degrade
        sample_rate_hz=100.0               # your sensor’s rate
    )


    X01_label = pd.read_csv(rf"/home/ws/ugoby/master_thesis/data/labels/{train_1}.csv")

    timeseries_tools.generate_images_fixed_count(X01_label, "acc_y_combined", method="mfcc", output_folder=rf"/home/ws/ugoby/master_thesis/data/{train_1}", target_count=33090)

    




    X01_label = pd.read_csv(rf"/home/ws/ugoby/master_thesis/data/labels/{train_2}.csv")

    timeseries_tools.generate_images_fixed_count(X01_label, "acc_y_combined", method="mfcc", output_folder=rf"/home/ws/ugoby/master_thesis/data/{train_2}", target_count=15717)

    X01_label = pd.read_csv(rf"/home/ws/ugoby/master_thesis/data/labels/{train_3}.csv")

    timeseries_tools.generate_images_fixed_count(X01_label, "acc_y_combined", method="mfcc", output_folder=rf"/home/ws/ugoby/master_thesis/data/{train_3}", target_count=15717)

    X01_label = pd.read_csv(rf"/home/ws/ugoby/master_thesis/data/labels/{train_4}.csv")

    timeseries_tools.generate_images_fixed_count(X01_label, "acc_y_combined", method="mfcc", output_folder=rf"/home/ws/ugoby/master_thesis/data/{train_4}", target_count=15717)

    X01_label = pd.read_csv(rf"/home/ws/ugoby/master_thesis/data/labels/{train_5}.csv")

    timeseries_tools.generate_images_fixed_count(X01_label, "acc_y_combined", method="mfcc", output_folder=rf"/home/ws/ugoby/master_thesis/data/{train_5}", target_count=15717)

    X01_label = pd.read_csv(rf"/home/ws/ugoby/master_thesis/data/labels/{train_6}.csv")

    timeseries_tools.generate_images_fixed_count(X01_label, "acc_y_combined", method="mfcc", output_folder=rf"/home/ws/ugoby/master_thesis/data/{train_6}", target_count=15717)


    

    
    ##only 1 backbone
    parameters = [
    {
        "image_folder": fr"/home/ws/ugoby/master_thesis/data/{train_1}",
        "csv_file": r"",
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "learning_rate": 1e-4,
        "batch_size": 32,
        "model_class": networks.ConvNeXtBaseClassifier,
        "resize": (224, 224),
        "epochs": 50,
        "writer": SummaryWriter(log_dir=fr"/home/ws/ugoby/master_thesis/tensorboard/runs/single_backbones\{train_1}"),
        "model_save_path": fr"/home/ws/ugoby/master_thesis/models/single_backbones/{train_1}.pt",
        "data_type": "mfcc",
        "num_classes": 3,  # 3 for normal , 6 for night

        # Early stopping additions:
        "early_stopping_patience": 10,
        "min_delta": 0.001
    },
    ]


'''