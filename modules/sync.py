import sys
import time
import concurrent.futures
import numpy as np
import scipy.signal
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
import soundfile as sf
from typing import Optional, Any

from .utils import (
    log_msg, draw_progress_bar, format_time, _save_audio_atomic
)
from .config import (
    DTW_RESOLUTION, SYNC_METHOD
)
from .hardware import (
    GPU_VRAM_GB, CPU_THREADS
)

# Optional imports for DTW
DTW_IMPORT_ERROR: Optional[str] = None
try:
    import librosa
    import fastdtw
except ImportError as e:
    DTW_IMPORT_ERROR = str(e)
    librosa = None  # type: ignore[no-redef]
    fastdtw = None  # type: ignore[no-redef]
else:
    DTW_IMPORT_ERROR = None


def _run_fastdtw_chunk(args):
    """
    Worker function for Parallel DTW.
    Args: (ref_segment, proc_segment, radius)
    Returns: path (list of [ref_idx, proc_idx])
    NOTE: 'dist' function (euclidean) must be imported in worker scope or passed.
    SciPy euclidean is picklable.
    """
    ref_seg, proc_seg, radius = args
    # Ensure dependencies in worker process
    import fastdtw
    from scipy.spatial.distance import euclidean

    _, path = fastdtw.fastdtw(ref_seg, proc_seg, radius=radius, dist=euclidean)
    return path


def _run_gpu_dtw_chunk(args):
    """
    Worker function for GPU-accelerated DTW (Hybrid).
    1. Compute Euclidean Distance Matrix on GPU (torch.cdist).
    2. Compute DTW Path on CPU (librosa.sequence.dtw).

    Args: (ref_segment, proc_segment, radius)
    Returns: path (list of [ref_idx, proc_idx])
    """
    ref_seg, proc_seg, radius = args

    # 1. GPU Distance Calculation
    import torch
    import librosa

    # Ensure tensors are on GPU
    # Note: ref_seg/proc_seg are numpy arrays (Features x Time) -> Need Transpose?
    # ref_seg passed in is (Time, Features) from `chunks.append`.
    # torch.cdist expects (B, P, M) or (P, M).

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_t = torch.tensor(ref_seg, dtype=torch.float32, device=device)
    proc_t = torch.tensor(proc_seg, dtype=torch.float32, device=device)

    # Compute Cost Matrix (N x M)
    # torch.cdist computes Euclidean distance
    cost_matrix = torch.cdist(ref_t, proc_t).cpu().numpy()

    # 2. CPU Path Finding (Librosa/Numba)
    # librosa.sequence.dtw expects 'C' as pre-computed cost matrix.
    # It returns D (cumulative cost) and wp (path).
    # wp is (N_steps, 2) in (row, col) -> (ref_idx, proc_idx)
    # Librosa returns path from end-to-start ([N-1, M-1] ... [0, 0]).

    _, wp = librosa.sequence.dtw(C=cost_matrix, global_constraints=False)

    # Reverse to get Start -> End ([0, 0] ... [N-1, M-1])
    wp = wp[::-1]

    # Convert to list of lists for compatibility
    return wp.tolist()


def _apply_warp_gpu(audio_np, source_indices_np):
    """
    Applies warping using GPU acceleration (torch.nn.functional.grid_sample).
    Audio: (samples, channels) numpy
    Indices: (output_samples,) numpy - The source index for each output sample.
    """
    import torch
    try:
        if not torch.cuda.is_available():
            return None

        device = torch.device("cuda")

        # Prepare Audio: (N, C, H, W) -> (1, Channels, 1, Samples)
        # Audio is currently (Samples, Channels) from soundfile
        # Transpose to (Channels, Samples)
        audio_t = torch.from_numpy(audio_np.T).float().unsqueeze(0).unsqueeze(2).to(device)

        # Normalize Indices to [-1, 1] for grid_sample
        # -1 = 0, +1 = MaxSample
        W_in = audio_np.shape[0]
        # Avoid division by zero
        if W_in <= 1:
            return None

        indices_t = torch.from_numpy(source_indices_np).float().to(device)
        grid_x = 2.0 * indices_t / (W_in - 1) - 1.0

        # Construct Grid: (N, H_out, W_out, 2)
        # Here H_out=1, W_out=LenIndices
        W_out = len(source_indices_np)

        # Stack (x, y) coordinates
        # y is always 0 (center of 1-pixel height)
        # Shape: (1, 1, W_out, 2)
        grid = torch.zeros(1, 1, W_out, 2, device=device)
        grid[0, 0, :, 0] = grid_x
        grid[0, 0, :, 1] = 0

        # Resample
        # align_corners=True ensures -1 maps to index 0 and 1 maps to index W-1
        warped = torch.nn.functional.grid_sample(
            audio_t, grid,
            mode='bicubic',
            padding_mode='zeros',
            align_corners=True
        )

        # Output: (1, C, 1, W_out)
        # Back to (W_out, C)
        warped_np = warped.squeeze(2).squeeze(0).permute(1, 0).cpu().numpy()
        return warped_np

    except Exception as e:
        log_msg(f"    [Warning] GPU Warp failed ({e}). Fallback to CPU.")
        return None


def _load_dtw_features(original_wav, processed_wav):
    """Loads audio and computes chroma features for DTW."""
    ANALYSIS_SR = 8192
    hop_length = int(ANALYSIS_SR / DTW_RESOLUTION)

    t0 = time.time()
    ref_y, _ = librosa.load(str(original_wav), sr=ANALYSIS_SR, mono=True)
    proc_y, _ = librosa.load(str(processed_wav), sr=ANALYSIS_SR, mono=True)
    log_msg(f"    Loaded for analysis in {time.time() - t0:.2f}s")

    audio_dur = len(ref_y) / ANALYSIS_SR

    ref_chroma = librosa.feature.chroma_stft(
        y=ref_y, sr=ANALYSIS_SR, hop_length=hop_length, n_fft=2048
    )
    proc_chroma = librosa.feature.chroma_stft(
        y=proc_y, sr=ANALYSIS_SR, hop_length=hop_length, n_fft=2048
    )

    return ref_chroma.T, proc_chroma.T, audio_dur, hop_length, ANALYSIS_SR


def _prepare_dtw_chunks(ref_features, proc_features):
    """Splits features into overlapping chunks."""
    radius = max(5, int(DTW_RESOLUTION * 0.3))

    # Chunk size: 3000 frames @ 40Hz = 75 seconds.
    CHUNK_SIZE = 3000
    OVERLAP = 100

    len_ref = len(ref_features)
    chunks = []
    chunk_starts = []

    for i in range(0, len_ref, CHUNK_SIZE - OVERLAP):
        start = i
        end = min(len_ref, i + CHUNK_SIZE)
        ref_seg = ref_features[start:end]
        proc_seg = proc_features[start:end]
        chunks.append((ref_seg, proc_seg, radius))
        chunk_starts.append(start)
        if end == len_ref:
            break

    return chunks, chunk_starts


def _execute_parallel_dtw(chunks):
    """Executes DTW chunks in parallel on CPU or GPU."""
    use_gpu_dtw = False
    try:
        import torch
        if torch.cuda.is_available() and GPU_VRAM_GB >= 4:
            use_gpu_dtw = True
    except Exception:
        pass

    if use_gpu_dtw:
        max_workers = min(len(chunks), 4)
        Worker = _run_gpu_dtw_chunk
        Executor = concurrent.futures.ThreadPoolExecutor
        log_msg(f"    GPU Optimization ENABLED (VRAM: {GPU_VRAM_GB}GB). Using Torch+Librosa.")
    else:
        max_workers = min(len(chunks), CPU_THREADS)
        Worker = _run_fastdtw_chunk
        Executor = concurrent.futures.ProcessPoolExecutor

    log_msg(f"    Spawning {max_workers} workers for {len(chunks)} chunks...")

    results_map = {}
    chunks_done = 0
    num_chunks = len(chunks)
    t0 = time.time()

    draw_progress_bar(0, "DTW Sync: Starting...")

    with Executor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(Worker, chunk): i
            for i, chunk in enumerate(chunks)
        }

        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                res = future.result()
                results_map[chunk_idx] = res
            except Exception as e:
                log_msg(f"    Chunk {chunk_idx} failed: {e}", is_error=True)
                raise e

            chunks_done += 1
            percent = (chunks_done / num_chunks) * 100
            elapsed = time.time() - t0
            draw_progress_bar(
                percent, f"DTW Sync: Chunk {chunks_done}/{num_chunks}",
                elapsed_sec=elapsed
            )

    path_segments = [results_map[i] for i in range(num_chunks)]
    sys.stdout.write("\n")
    return path_segments


def _stitch_dtw_path(path_segments, chunk_starts):
    """Stitches chunked paths into a single monotonic path."""
    full_path = []

    for i, segment_path in enumerate(path_segments):
        base_idx = chunk_starts[i]
        seg_arr = np.array(segment_path)
        seg_arr += base_idx
        full_path.append(seg_arr)

    path = np.vstack(full_path)

    # Sort by Ref Index (Time)
    sort_order = np.argsort(path[:, 0])
    path = path[sort_order]

    # Unique Ref indices
    _, unique_mask = np.unique(path[:, 0], return_index=True)
    path = path[unique_mask]

    # Enforce Monotonicity on Proc Indices
    path[:, 1] = np.maximum.accumulate(path[:, 1])

    log_msg(f"    Stitched segments into {len(path)} points.")
    return path


def _warp_aligned_audio_cpu(full_proc_audio, source_indices, num_channels, output_wav, full_sr):
    """Fallback CPU warping."""
    log_msg("    GPU Warp unavailable/failed. Using CPU (scipy)...")
    out_audio = np.zeros_like(full_proc_audio)
    for ch in range(num_channels):
        out_audio[:, ch] = map_coordinates(
            full_proc_audio[:, ch],
            [source_indices],
            order=3
        )
    _save_audio_atomic(output_wav, out_audio, full_sr, subtype='FLOAT')


def _warp_aligned_audio(processed_wav, output_wav, path, ref_features_len, proc_features_len):
    """Warps the processed audio to align with reference."""

    # Coordinate Mapping Function
    ref_indices = path[:, 0]
    proc_indices = path[:, 1]
    _, unique_idxs = np.unique(ref_indices, return_index=True)
    unique_idxs.sort()
    target_t = ref_indices[unique_idxs]
    source_t = proc_indices[unique_idxs]

    warp_func = interp1d(target_t, source_t, kind='linear', fill_value="extrapolate")

    full_proc_audio, full_sr = sf.read(str(processed_wav), always_2d=True)
    num_frames = len(full_proc_audio)
    num_channels = full_proc_audio.shape[1]

    max_ref_idx = ref_features_len - 1
    max_proc_idx = proc_features_len - 1

    # Map: Full Sample Index -> Feature Index
    grid_in_feature_domain = np.linspace(0, max_ref_idx, num_frames)
    warped_feature_indices = warp_func(grid_in_feature_domain)

    # Smoothing
    window_length = 51  # minimal default
    if len(warped_feature_indices) > window_length:
        try:
            from scipy.signal import savgol_filter
            warped_feature_indices = savgol_filter(warped_feature_indices, window_length, 3)
        except Exception:
            pass

    # Feature Index -> Full Sample Index
    source_indices = warped_feature_indices * (num_frames / max_proc_idx)

    t0 = time.time()
    warped_gpu = _apply_warp_gpu(full_proc_audio, source_indices)

    if warped_gpu is not None:
        log_msg(f"    GPU Warping complete in {time.time() - t0:.2f}s")
        _save_audio_atomic(output_wav, warped_gpu, full_sr, subtype='FLOAT')
    else:
        _warp_aligned_audio_cpu(full_proc_audio, source_indices, num_channels, output_wav, full_sr)

    draw_progress_bar(100, "Sync: Complete")
    sys.stdout.write("\n")


def _calculate_cross_correlation_lag(ref_audio, proc_audio, sr):
    """Calculates lag using cross correlation."""
    # Convert stereo to mono for correlation
    if len(ref_audio.shape) > 1:
        ref_audio = np.mean(ref_audio, axis=1)
    if len(proc_audio.shape) > 1:
        proc_audio = np.mean(proc_audio, axis=1)

    draw_progress_bar(50, "Sync: Calculating Correlation...")
    correlation = scipy.signal.correlate(ref_audio, proc_audio, mode='full', method='fft')
    lags = scipy.signal.correlation_lags(len(ref_audio), len(proc_audio), mode='full')
    lag = lags[np.argmax(correlation)]

    log_msg(f"    Detected Lag: {lag} samples ({lag / sr * 1000:.2f} ms)")
    return lag


def _apply_shift_to_audio(processed_wav, output_wav, lag):
    """Applies shift to audio and saves it."""
    draw_progress_bar(80, "Sync: Applying Shift...")

    proc_audio, proc_sr = sf.read(str(processed_wav), always_2d=True)
    if proc_audio.shape[1] == 1:
        proc_audio = np.tile(proc_audio, (1, 2))

    shift_samples = -lag

    if shift_samples != 0:
        log_msg(f"    Applying shift: {shift_samples} samples")
        shifted_audio = np.roll(proc_audio, shift_samples, axis=0)

        # Zero out wrapped around part to avoid artifacts
        if shift_samples > 0:
            shifted_audio[:shift_samples] = 0
        else:
            shifted_audio[shift_samples:] = 0
    else:
        log_msg("    No shift needed.")
        shifted_audio = proc_audio

    # Atomic Write
    if not _save_audio_atomic(output_wav, shifted_audio, proc_sr, subtype='FLOAT'):
        raise Exception(f"Alignment Failed: Could not save shifted audio to {output_wav}")


def _align_stems_shift(original_wav, processed_wav, output_wav):
    """
    Step 4: Smart Audio Sync via Cross-Correlation.
    Calculates the lag between original and processed audio and shifts the processed
    audio to match the original timing perfectly.
    """
    log_msg(f"    Shift Sync: Aligning {processed_wav.name}...")

    try:
        draw_progress_bar(10, "Sync: Loading Audio...")
        ref_audio, sr = sf.read(str(original_wav), frames=44100 * 60)
        proc_audio, _ = sf.read(str(processed_wav), frames=44100 * 60)

        draw_progress_bar(30, "Sync: Preparing Stems...")

        n = len(ref_audio)
        m = len(proc_audio)
        if n == 0 or m == 0:
            log_msg("    [Warning] Audio empty, skipping sync.", is_error=True)
            data, rate = sf.read(str(processed_wav), always_2d=True)
            if data.shape[1] == 1:
                data = np.tile(data, (1, 2))
            sf.write(str(output_wav), data, rate, subtype='FLOAT')
            draw_progress_bar(100, "Sync: Skipped (Empty)")
            sys.stdout.write("\n")
            return output_wav

        lag = _calculate_cross_correlation_lag(ref_audio, proc_audio, sr)
        _apply_shift_to_audio(processed_wav, output_wav, lag)

        draw_progress_bar(100, "Sync: Complete")
        sys.stdout.write("\n")
        return output_wav

    except Exception as e:
        log_msg(f"    [Warning] Sync failed ({e}). Using unaligned.", is_error=True)
        try:
            data, rate = sf.read(str(processed_wav), always_2d=True)
            if data.shape[1] == 1:
                data = np.tile(data, (1, 2))
            sf.write(str(output_wav), data, rate, subtype='FLOAT')
            draw_progress_bar(100, "Sync: Skipped (Fallback)")
        except Exception:
            pass
        return output_wav


def _align_stems_dtw(original_wav, processed_wav, output_wav):
    """
    Advanced Sync: Dynamic Time Warping (DTW).
    Corrects variable drift (wow/flutter) by warping the processed audio
    to match the original timing perfectly.
    """
    if librosa is None or fastdtw is None:
        msg = "[Warning] DTW dependencies missing. Falling back to Shift."
        if DTW_IMPORT_ERROR:
            msg += f" (Error: {DTW_IMPORT_ERROR})"
        log_msg(msg, is_error=True)
        return _align_stems_shift(original_wav, processed_wav, output_wav)

    log_msg(f"    DTW Sync: Warping {processed_wav.name}...")

    try:
        ref_features, proc_features, audio_dur, _, _ = _load_dtw_features(original_wav, processed_wav)

        res_factor = DTW_RESOLUTION / 100.0
        est_dtw_time = audio_dur * 0.06 * res_factor
        log_msg(f"    Computing DTW Path (Est. Time: {format_time(est_dtw_time)})...")

        chunks, chunk_starts = _prepare_dtw_chunks(ref_features, proc_features)

        path_segments = _execute_parallel_dtw(chunks)
        log_msg("    DTW Path computed, stitching...")

        path = _stitch_dtw_path(path_segments, chunk_starts)

        _warp_aligned_audio(processed_wav, output_wav, path, len(ref_features), len(proc_features))

        return output_wav

    except Exception as e:
        log_msg(f"    [Warning] DTW Sync failed ({e}). Falling back to Shift.", is_error=True)
        return _align_stems_shift(original_wav, processed_wav, output_wav)


def _align_stems(original_wav, processed_wav, output_wav):
    if SYNC_METHOD == "dtw":
        return _align_stems_dtw(original_wav, processed_wav, output_wav)
    else:
        return _align_stems_shift(original_wav, processed_wav, output_wav)
