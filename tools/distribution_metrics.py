"""
@filename distribution_metrics.py

This script provides a class `DistributionMetrics` to compute and visualize distribution similarity metrics 
between training and validation datasets stored in HDF5 files. It supports the following functionalities:

1. **Loading Dataset Statistics**: Computes basic statistics (mean, standard deviation, histograms) for each channel 
   in the datasets.
2. **Computing Metrics**:
   - Wasserstein Distance
   - Jensen-Shannon Divergence
   - Maximum Mean Discrepancy (MMD)
3. **Visualization**:
   - Channel-wise pixel value distributions
   - Comparison of statistical moments (mean, standard deviation)
   - Heatmap of distribution similarity metrics

Usage:
- Initialize the `DistributionMetrics` class with paths to the training and validation HDF5 files.
- Call `compute_all_metrics` to compute and visualize the metrics.
"""

import numpy as np
import h5py
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class DistributionMetrics:
    """
    Computes and visualizes distribution similarity metrics between training and validation datasets.
    """
    
    CHANNEL_NAMES = ['Red', 'Green', 'Blue']
    COLORS = ['red', 'green', 'blue']
    
    def __init__(self, train_h5_path: str, val_h5_path: str, output_dir: Optional[str] = None):
        """
        Initialize with paths to training and validation HDF5 files.
        
        Args:
            train_h5_path (str): Path to training HDF5 file
            val_h5_path (str): Path to validation HDF5 file
            output_dir (Optional[str]): Directory to save visualization plots
        """
        self.train_h5_path = train_h5_path
        self.val_h5_path = val_h5_path
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "distribution_plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def _load_dataset_statistics(self, h5_path: str, max_samples: Optional[int] = None) -> Dict:
        """
        Load and compute basic statistics from an HDF5 dataset.
        
        Args:
            h5_path (str): Path to HDF5 file
            max_samples (Optional[int]): Maximum number of samples to process
            
        Returns:
            Dict: Dictionary containing dataset statistics
        """
        stats = {
            'means': [],
            'stds': [],
            'histograms': [],
            'samples': []
        }
        
        with h5py.File(h5_path, 'r') as h5f:
            keys = list(h5f.keys())
            if max_samples:
                keys = keys[:max_samples]
                
            for key in tqdm(keys, desc=f"Processing {h5_path}"):
                data = h5f[key][:] # type: ignore
                
                # Store basic statistics
                stats['means'].append(np.mean(data, axis=(1,2))) # type: ignore
                stats['stds'].append(np.std(data, axis=(1,2))) # type: ignore
                
                # Compute histogram for each channel
                for channel in range(data.shape[0]): # type: ignore
                    hist, _ = np.histogram(data[channel], bins=50, range=(0, 1), density=True) # type: ignore
                    stats['histograms'].append(hist)
                
                # Store flattened samples for MMD computation
                stats['samples'].append(data.reshape(data.shape[0], -1)) # type: ignore
                
        return stats
    
    def compute_wasserstein(self, train_hist: np.ndarray, val_hist: np.ndarray) -> float:
        """
        Compute Wasserstein distance between two distributions.
        """
        # Normalize histograms to ensure they sum to 1
        train_hist = train_hist / np.sum(train_hist)
        val_hist = val_hist / np.sum(val_hist)
        
        # Generate bin centers (assuming bins are evenly spaced in [0,1])
        bins = np.linspace(0, 1, len(train_hist))
        
        return wasserstein_distance(bins, bins, train_hist, val_hist)
    
    def compute_jensen_shannon(self, train_hist: np.ndarray, val_hist: np.ndarray) -> float:
        """
        Compute Jensen-Shannon divergence between two distributions.
        """
        # Normalize histograms
        train_hist = train_hist / np.sum(train_hist)
        val_hist = val_hist / np.sum(val_hist)
        
        return float(jensenshannon(train_hist, val_hist))
    
    def compute_mmd(self, X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
        """
        Compute Maximum Mean Discrepancy with RBF kernel.
        
        Args:
            X (np.ndarray): First sample set
            Y (np.ndarray): Second sample set
            gamma (float): RBF kernel bandwidth parameter
            
        Returns:
            float: MMD value
        """
        def rbf_kernel(X, Y, gamma):
            """RBF kernel computation"""
            X_norm = np.sum(X**2, axis=1)
            Y_norm = np.sum(Y**2, axis=1)
            gamma2 = -gamma/2.0
            K = np.exp(gamma2 * (X_norm[:,None] + Y_norm[None,:] - 2 * np.dot(X, Y.T)))
            return K
            
        Kxx = rbf_kernel(X, X, gamma)
        Kyy = rbf_kernel(Y, Y, gamma)
        Kxy = rbf_kernel(X, Y, gamma)
        
        mx = Kxx.mean()
        my = Kyy.mean()
        mxy = Kxy.mean()
        
        return mx + my - 2*mxy
    
    def plot_channel_distributions(self, train_stats: Dict, val_stats: Dict) -> None:
        """
        Plot the distribution of pixel values for each channel comparing train and validation sets.
        
        Args:
            train_stats (Dict): Training set statistics
            val_stats (Dict): Validation set statistics
        """
        train_means = np.array(train_stats['means'])
        val_means = np.array(val_stats['means'])
        
        # Create figure with subplots for each channel
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Channel-wise Pixel Value Distributions: Training vs Validation')
        
        for idx, (channel_name, color) in enumerate(zip(self.CHANNEL_NAMES, self.COLORS)):
            ax = axes[idx]
            
            # Plot training distribution
            sns.kdeplot(train_means[:, idx], ax=ax, color=color, 
                       label='Training', alpha=0.6)
            # Plot validation distribution
            sns.kdeplot(val_means[:, idx], ax=ax, color='gray',
                       label='Validation', alpha=0.6, linestyle='--')
            
            ax.set_title(f'{channel_name} Channel')
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Density')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'channel_distributions.png')
        plt.close()
    
    def plot_moment_comparisons(self, metrics: Dict) -> None:
        """
        Plot comparison of statistical moments between training and validation sets.
        
        Args:
            metrics (Dict): Dictionary containing computed metrics
        """
        moment_diffs = metrics['moment_differences']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x = np.arange(len(moment_diffs))
        bars = ax.bar(x, moment_diffs.values())
        
        ax.set_title('Differences in Statistical Moments\nBetween Training and Validation Sets')
        ax.set_xticks(x)
        ax.set_xticklabels(moment_diffs.keys())
        ax.set_ylabel('Absolute Difference')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'moment_differences.png')
        plt.close()
    
    def plot_metric_heatmap(self, metrics: Dict) -> None:
        """
        Create a heatmap visualization of the distribution metrics.
        
        Args:
            metrics (Dict): Dictionary containing computed metrics
        """
        # Prepare data for heatmap
        metric_data = {
            'Wasserstein': np.mean(metrics['wasserstein']),
            'Jensen-Shannon': np.mean(metrics['jensen_shannon']),
            'MMD': metrics['mmd']
        }
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Reshape data for heatmap
        data_array = np.array(list(metric_data.values())).reshape(1, -1)
        
        sns.heatmap(data_array, annot=True, fmt='.4f', cmap='YlOrRd',
                   xticklabels=list(metric_data.keys()),
                   yticklabels=['Train vs Val'],
                   ax=ax)
        
        ax.set_title('Distribution Similarity Metrics Overview')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_heatmap.png')
        plt.close()

    def compute_all_metrics(self, max_samples: Optional[int] = None) -> Dict:
        """
        Compute all distribution similarity metrics between training and validation sets.
        
        Args:
            max_samples (Optional[int]): Maximum number of samples to process
            
        Returns:
            Dict: Dictionary containing all computed metrics
        """
        # Load dataset statistics
        train_stats = self._load_dataset_statistics(self.train_h5_path, max_samples)
        val_stats = self._load_dataset_statistics(self.val_h5_path, max_samples)
        
        metrics = {
            'wasserstein': [],
            'jensen_shannon': [],
            'mmd': None,
            'moment_differences': {
                'mean': [],
                'std': []
            }
        }
        
        # Compute histogram-based metrics
        for train_hist, val_hist in zip(train_stats['histograms'], val_stats['histograms']):
            metrics['wasserstein'].append(self.compute_wasserstein(train_hist, val_hist))
            metrics['jensen_shannon'].append(self.compute_jensen_shannon(train_hist, val_hist))
        
        # Compute MMD on a subset of samples
        train_samples = np.vstack(train_stats['samples'][:100])  # Limit samples for computational efficiency
        val_samples = np.vstack(val_stats['samples'][:100])
        metrics['mmd'] = self.compute_mmd(train_samples, val_samples)
        
        # Compute moment differences
        train_means = np.array(train_stats['means'])
        val_means = np.array(val_stats['means'])
        train_stds = np.array(train_stats['stds'])
        val_stds = np.array(val_stats['stds'])
        
        # Compare distribution statistics rather than element-wise differences
        metrics['moment_differences']['mean'] = np.mean(np.abs(
            np.mean(train_means, axis=0) - np.mean(val_means, axis=0)
        ))
        metrics['moment_differences']['std'] = np.mean(np.abs(
            np.mean(train_stds, axis=0) - np.mean(val_stds, axis=0)
        ))
        
        # Generate visualizations
        self.plot_channel_distributions(train_stats, val_stats)
        self.plot_moment_comparisons(metrics)
        self.plot_metric_heatmap(metrics)
        
        return metrics

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize metrics calculator
    calculator = DistributionMetrics(
        train_h5_path='data/train_color.h5',
        val_h5_path='data/val_color.h5',
        output_dir='output/train_val_distribution_plots'
    )
    
    # Compute metrics
    metrics = calculator.compute_all_metrics(max_samples=1000)
    
    # Print results
    print("\nDistribution Similarity Metrics:")
    print(f"Wasserstein Distance (mean): {np.mean(metrics['wasserstein']):.4f}")
    print(f"Jensen-Shannon Divergence (mean): {np.mean(metrics['jensen_shannon']):.4f}")
    print(f"Maximum Mean Discrepancy: {metrics['mmd']:.4f}")
    print("\nMoment Differences:")
    print(f"Mean Difference: {metrics['moment_differences']['mean']:.4f}")
    print(f"Std Difference: {metrics['moment_differences']['std']:.4f}")