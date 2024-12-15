import time
from typing import Callable, Tuple
import numpy as np
from numba import cuda
from utils.image import srgb_to_linear as stl, linear_to_srgb as lts

# Core parameters for sRGB to linear and linear to sRGB conversion
# These numbers are derived from the sRGB standard (IEC 61966-2-1:1999), used for gamma correction in color spaces.
COLOR_SPACE_PARAMS = {
    "srgb_to_linear": {
        "threshold": 0.04045,  # Transition point between linear and non-linear scaling in sRGB.
        "scale_factor": 12.92, # Scaling factor for linear colors below the threshold.
        "gamma": 2.4,          # Gamma correction exponent for non-linear colors.
        "offset": 0.055,       # Offset added to normalize non-linear scaling.
        "multiplier": 1.055,   # Multiplier for non-linear scaling adjustment.
    },
    "linear_to_srgb": {
        "threshold": 0.0031308, # Transition point between linear and non-linear scaling in linear space.
        "scale_factor": 12.92,  # Scaling factor for linear colors below the threshold.
        "gamma": 2.4,           # Inverse gamma correction for non-linear colors.
        "offset": 0.055,        # Offset to normalize sRGB scaling.
        "multiplier": 1.055,    # Multiplier to match the sRGB curve.
    },
}

@cuda.jit
def srgb_to_linear_cuda(color, result, threshold, scale_factor, gamma, offset, multiplier):
    # Flattened grid for better memory coalescing
    idx = cuda.grid(1) # type: ignore
    total_pixels = color.shape[0] * color.shape[1]
    
    if idx < total_pixels:
        y = idx // color.shape[1]
        x = idx % color.shape[1]
        for c in range(color.shape[2]):
            value = color[y, x, c]
            if value <= threshold:
                result[y, x, c] = value / scale_factor
            else:
                result[y, x, c] = ((value + offset) / multiplier) ** gamma

@cuda.jit
def linear_to_srgb_cuda(color, result, threshold, scale_factor, gamma, offset, multiplier):
    # Flattened grid for better memory coalescing
    idx = cuda.grid(1) # type: ignore
    total_pixels = color.shape[0] * color.shape[1]
    
    if idx < total_pixels:
        y = idx // color.shape[1]
        x = idx % color.shape[1]
        for c in range(color.shape[2]):
            value = color[y, x, c]
            if value <= threshold:
                result[y, x, c] = value * scale_factor
            else:
                result[y, x, c] = multiplier * (value ** (1 / gamma)) - offset

def get_optimal_block_size():
    # Standard block size (block_size_x, block_size_y)
    return (16, 16)

def srgb_to_linear(color: np.ndarray, randomize: bool = False) -> np.ndarray:
    if color.ndim != 3:
        raise ValueError(f"Input color array must be 3D (H, W, C), but got shape {color.shape}")
    
    params = (
        introduce_random_variation(COLOR_SPACE_PARAMS["srgb_to_linear"])
        if randomize
        else COLOR_SPACE_PARAMS["srgb_to_linear"]
    )

    # Ensure contiguous memory layout and float32 data type
    color = np.ascontiguousarray(color, dtype=np.float32)
    
    # Allocate device memory
    d_color = cuda.to_device(color)
    d_result = cuda.device_array_like(color)

    # Calculate optimal block and grid sizes
    threads_per_block = 256
    total_pixels = color.shape[0] * color.shape[1]
    blocks_per_grid = (total_pixels + threads_per_block - 1) // threads_per_block

    # Launch kernel
    srgb_to_linear_cuda[blocks_per_grid, threads_per_block](d_color, d_result, # type: ignore
                                                             params["threshold"],
                                                             params["scale_factor"],
                                                             params["gamma"],
                                                             params["offset"],
                                                             params["multiplier"])
    # Ensure kernel execution is complete
    cuda.synchronize()

    # Copy result back to host
    result = d_result.copy_to_host()

    return result

def linear_to_srgb(color: np.ndarray, randomize: bool = False) -> np.ndarray:
    if color.ndim != 3:
        raise ValueError(f"Input color array must be 3D (H, W, C), but got shape {color.shape}")
    
    params = (
        introduce_random_variation(COLOR_SPACE_PARAMS["linear_to_srgb"])
        if randomize
        else COLOR_SPACE_PARAMS["linear_to_srgb"]
    )

    # Ensure contiguous memory layout and float32 data type
    color = np.ascontiguousarray(color, dtype=np.float32)
    
    # Allocate device memory
    d_color = cuda.to_device(color)
    d_result = cuda.device_array_like(color)

    # Calculate optimal block and grid sizes
    threads_per_block = 256
    total_pixels = color.shape[0] * color.shape[1]
    blocks_per_grid = (total_pixels + threads_per_block - 1) // threads_per_block

    # Launch kernel
    linear_to_srgb_cuda[blocks_per_grid, threads_per_block](d_color, d_result, # type: ignore
                                                             params["threshold"],
                                                             params["scale_factor"],
                                                             params["gamma"],
                                                             params["offset"],
                                                             params["multiplier"])
    # Ensure kernel execution is complete
    cuda.synchronize()

    # Copy result back to host
    result = d_result.copy_to_host()

    return result

def introduce_random_variation(base_params: dict) -> dict:
    # Introduce small random variations to parameters if needed
    random_values = np.random.uniform(
        low=[-0.01, -0.1, -0.05, -0.01, -0.01],
        high=[0.01, 0.1, 0.05, 0.01, 0.01],
        size=5
    )
    
    return {
        "threshold": base_params["threshold"] + random_values[0],
        "scale_factor": base_params["scale_factor"] + random_values[1],
        "gamma": base_params["gamma"] + random_values[2],
        "offset": base_params["offset"] + random_values[3],
        "multiplier": base_params["multiplier"] + random_values[4],
    }

def benchmark_function(func: Callable, data: np.ndarray, warmup_runs: int = 3, test_runs: int = 10) -> Tuple[float, float]:
    """
    Benchmark a function with given data
    Returns: (mean_time, std_dev)
    """
    # Warmup runs to ensure JIT compilation and caching
    for _ in range(warmup_runs):
        _ = func(data)
    
    # Actual benchmark
    times = []
    
    for _ in range(test_runs):
        start_time = time.perf_counter()
        _ = func(data)
        cuda.synchronize()  # Ensure all CUDA operations are complete
        end_time = time.perf_counter()
        
        times.append(end_time - start_time)
    
    return np.mean(times), np.std(times) # type: ignore

if __name__ == "__main__":
    # Test with realistic image dimensions
    H, W, C = 1080, 1920, 3  # Example dimensions (e.g., 1080p image with 3 channels)
    test_colors = np.random.rand(H, W, C).astype(np.float32)

    max_difference_cuda = 0
    max_difference_numpy = 0
    iterations = 10

    for _ in range(iterations):
        # Generate random colors
        test_colors = np.random.rand(H, W, C).astype(np.float32)
        
        # CUDA-based conversion
        linear_colors_cuda = srgb_to_linear(test_colors, randomize=False)
        restored_colors_cuda = linear_to_srgb(linear_colors_cuda, randomize=False)
        current_max_diff_cuda = np.max(np.abs(test_colors - restored_colors_cuda))
        max_difference_cuda = max(max_difference_cuda, current_max_diff_cuda)
        
        # Numpy-based conversion
        linear_colors_numpy = stl(test_colors, randomize=False)
        restored_colors_numpy = lts(linear_colors_numpy, randomize=False)
        current_max_diff_numpy = np.max(np.abs(test_colors - restored_colors_numpy))
        max_difference_numpy = max(max_difference_numpy, current_max_diff_numpy)

    # Benchmarking CUDA functions
    benchmark_runs = 100  # Reduced to a reasonable number for demonstration
    mean_time_stl_cuda, std_dev_stl = benchmark_function(srgb_to_linear, test_colors, test_runs=benchmark_runs)
    mean_time_lts_cuda, std_dev_lts = benchmark_function(linear_to_srgb, test_colors, test_runs=benchmark_runs)

    print("== CUDA-Version ==")
    print("Original shape:", test_colors.shape)
    print("Max difference:", max_difference_cuda)
    print(f"Mean time srgb_to_linear: {mean_time_stl_cuda:.6f} ± {std_dev_stl:.6f} seconds")
    print(f"Mean time linear_to_srgb: {mean_time_lts_cuda:.6f} ± {std_dev_lts:.6f} seconds")
    print("\n")

    # Benchmarking Numpy functions
    mean_time_stl_numpy, std_dev_stl_numpy = benchmark_function(stl, test_colors, warmup_runs=3, test_runs=10)
    mean_time_lts_numpy, std_dev_lts_numpy = benchmark_function(lts, test_colors, warmup_runs=3, test_runs=10)

    print("== Numpy-Version ==")
    print("Original shape:", test_colors.shape)
    print("Max difference:", max_difference_numpy)
    print(f"Mean time srgb_to_linear: {mean_time_stl_numpy:.6f} ± {std_dev_stl_numpy:.6f} seconds")
    print(f"Mean time linear_to_srgb: {mean_time_lts_numpy:.6f} ± {std_dev_lts_numpy:.6f} seconds")
    print("\n")

    print("== Speedup ==")
    if mean_time_stl_cuda > 0:
        speedup_stl = mean_time_stl_numpy / mean_time_stl_cuda
        print(f"Speedup srgb_to_linear: {speedup_stl:.2f}x")
    else:
        print("Speedup srgb_to_linear: Infinity (CUDA time is zero)")

    if mean_time_lts_cuda > 0:
        speedup_lts = mean_time_lts_numpy / mean_time_lts_cuda
        print(f"Speedup linear_to_srgb: {speedup_lts:.2f}x")
    else:
        print("Speedup linear_to_srgb: Infinity (CUDA time is zero)")
