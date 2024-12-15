import gc
import time
from typing import Callable, Tuple
import numpy as np
from numba import cuda
from utils.image import srgb_to_linear as stl, linear_to_srgb as lts

COLOR_SPACE_PARAMS = {
    "srgb_to_linear": {
        "threshold": 0.04045,
        "scale_factor": 12.92,
        "gamma": 2.4,
        "offset": 0.055,
        "multiplier": 1.055,
    },
    "linear_to_srgb": {
        "threshold": 0.0031308,
        "scale_factor": 12.92,
        "gamma": 2.4,
        "offset": 0.055,
        "multiplier": 1.055,
    },
}

@cuda.jit
def srgb_to_linear_cuda(color, result, threshold, scale_factor, gamma, offset, multiplier):
    # Use 2D grid for better memory access patterns
    x, y = cuda.grid(2) # type: ignore
    height, width, channels = color.shape
    
    if x < width and y < height:
        for c in range(channels):
            idx = y * width * channels + x * channels + c
            if color[y, x, c] <= threshold:
                result[y, x, c] = color[y, x, c] / scale_factor
            else:
                result[y, x, c] = ((color[y, x, c] + offset) / multiplier) ** gamma

@cuda.jit
def linear_to_srgb_cuda(color, result, threshold, scale_factor, gamma, offset, multiplier):
    x, y = cuda.grid(2) # type: ignore
    height, width, channels = color.shape
    
    if x < width and y < height:
        for c in range(channels):
            if color[y, x, c] <= threshold:
                result[y, x, c] = color[y, x, c] * scale_factor
            else:
                result[y, x, c] = multiplier * (color[y, x, c] ** (1 / gamma)) - offset

def get_optimal_block_size(width, height):
    # Optimize block size based on image dimensions
    block_size_x = min(32, width)  # 32 is a good balance for memory access
    block_size_y = min(16, height)  # 16 rows per block works well for most cases
    return (block_size_x, block_size_y)

def srgb_to_linear(color: np.ndarray, randomize: bool = True) -> np.ndarray:
    if len(color.shape) != 3:
        color = color.reshape(-1, 4)  # Reshape to (H, W, C) if flattened
    
    params = (
        introduce_random_variation(COLOR_SPACE_PARAMS["srgb_to_linear"])
        if randomize
        else COLOR_SPACE_PARAMS["srgb_to_linear"]
    )

    # Ensure contiguous memory layout
    color = np.ascontiguousarray(color)
    
    # Allocate device memory
    d_color = cuda.to_device(color)
    d_result = cuda.device_array_like(color)

    # Calculate optimal block and grid sizes
    block_dim = get_optimal_block_size(color.shape[1], color.shape[0])
    grid_dim = (
        (color.shape[1] + block_dim[0] - 1) // block_dim[0],
        (color.shape[0] + block_dim[1] - 1) // block_dim[1]
    )

    # Launch kernel with 2D grid
    srgb_to_linear_cuda[grid_dim, block_dim]( # type: ignore
        d_color, d_result,
        params["threshold"],
        params["scale_factor"],
        params["gamma"],
        params["offset"],
        params["multiplier"]
    )

    # Use stream to overlap memory transfers
    stream = cuda.stream()
    result = d_result.copy_to_host(stream=stream)
    stream.synchronize()

    return result

def linear_to_srgb(color: np.ndarray, randomize: bool = True) -> np.ndarray:
    if len(color.shape) != 3:
        color = color.reshape(-1, 4)  # Reshape to (H, W, C) if flattened
    
    params = (
        introduce_random_variation(COLOR_SPACE_PARAMS["linear_to_srgb"])
        if randomize
        else COLOR_SPACE_PARAMS["linear_to_srgb"]
    )

    # Ensure contiguous memory layout
    color = np.ascontiguousarray(color)
    
    # Allocate device memory
    d_color = cuda.to_device(color)
    d_result = cuda.device_array_like(color)

    # Calculate optimal block and grid sizes
    block_dim = get_optimal_block_size(color.shape[1], color.shape[0])
    grid_dim = (
        (color.shape[1] + block_dim[0] - 1) // block_dim[0],
        (color.shape[0] + block_dim[1] - 1) // block_dim[1]
    )

    # Launch kernel with 2D grid
    linear_to_srgb_cuda[grid_dim, block_dim]( # type: ignore
        d_color, d_result,
        params["threshold"],
        params["scale_factor"],
        params["gamma"],
        params["offset"],
        params["multiplier"]
    )

    # Use stream to overlap memory transfers
    stream = cuda.stream()
    result = d_result.copy_to_host(stream=stream)
    stream.synchronize()

    return result

def introduce_random_variation(base_params: dict) -> dict:
    # Pre-calculate random values to reduce function calls
    random_values = np.random.uniform(
        [-0.01, -1.0, -0.3, -0.01, -0.01],
        [0.01, 1.0, 0.3, 0.01, 0.01],
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
    # Warmup runs to ensure JIT compilation
    for _ in range(warmup_runs):
        _ = func(data)
    
    # Actual benchmark
    times = []
    
    for _ in range(test_runs):
        gc.collect()  # Clear any garbage before each run
        
        start_time = time.perf_counter()
        _ = func(data)
        end_time = time.perf_counter()
        
        times.append(end_time - start_time)
    
    return np.mean(times), np.std(times) # type: ignore

if __name__ == "__main__":
    # Test with realistic image dimensions
    H, W, C = 1080, 640, 4  # Example dimensions
    test_colors = np.random.rand(H, W, C).astype(np.float32)

    max_difference_cuda = 0
    max_difference_numpy = 0
    for _ in range(10):
        test_colors = np.random.rand(H, W, C).astype(np.float32)
        linear_colors = srgb_to_linear(test_colors, randomize=False)
        restored_colors = linear_to_srgb(linear_colors, randomize=False)
        max_difference_cuda = np.max(np.abs(test_colors - restored_colors))
        linear_colors = stl(test_colors, randomize=False)
        restored_colors = lts(linear_colors, randomize=False)
        max_difference_numpy = np.max(np.abs(test_colors - restored_colors))


    mean_time_stl_cuda, std_dev_stl = benchmark_function(srgb_to_linear, test_colors, test_runs=200)
    mean_time_lts_cuda, std_dev_lts = benchmark_function(linear_to_srgb, test_colors, test_runs=200)

    print("== CUDA-Version ==")
    print("Original shape:", test_colors.shape)
    print("Max difference:", max_difference_cuda)
    print(f"Mean time stl: {mean_time_stl_cuda:.6f} ± {std_dev_stl:.6f} seconds")
    print(f"Mean time lts: {mean_time_lts_cuda:.6f} ± {std_dev_lts:.6f} seconds")
    print("\n")

    print("== Numpy-Version ==")
    mean_time_stl_numpy, std_dev_stl = benchmark_function(stl, test_colors, test_runs=1)
    mean_time_lts_numpy, std_dev_lts = benchmark_function(lts, test_colors, test_runs=1)
    print("Original shape:", test_colors.shape)
    print("Max difference:", max_difference_numpy)
    print(f"Mean time stl: {mean_time_stl_numpy:.6f} ± {std_dev_stl:.6f} seconds")
    print(f"Mean time lts: {mean_time_lts_numpy:.6f} ± {std_dev_lts:.6f} seconds")
    print("\n")

    print("== Speedup ==")
    print(f"Speedup stl: {mean_time_stl_numpy / mean_time_stl_cuda:.2f}x")
    print(f"Speedup lts: {mean_time_lts_numpy / mean_time_lts_cuda:.2f}x")
