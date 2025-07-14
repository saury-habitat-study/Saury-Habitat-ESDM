import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ===================================================================
# 1. Configuration and Constants
# ===================================================================

# Define the output folder for the plot
OUTPUT_FOLDER = "performance_plots"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Define the key performance metrics to visualize
METRICS_TO_PLOT = ['sensitivity', 'specificity', 'calibration']


# ===================================================================
# 2. Core Plotting Function
# ===================================================================

def plot_performance_pairplot(data_path):
    """
    Loads model performance data and generates a pairplot to visualize
    the distributions of and relationships between key metrics,
    differentiated by algorithm.

    Args:
        data_path (str): The path to the CSV file containing the performance data.
    """
    try:
        df = pd.read_csv(data_path)
        print(f"Performance data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Check if required columns exist in the dataframe
    required_cols = METRICS_TO_PLOT + ['algo']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Data must contain the columns: {required_cols}")
        return

    print("Generating pairplot...")

    # Create the pairplot using seaborn
    g = sns.pairplot(
        df,
        vars=METRICS_TO_PLOT,
        hue='algo',  # Color points and distributions by algorithm
        diag_kind='kde',  # Use Kernel Density Estimate for diagonal plots
        plot_kws={'alpha': 0.7, 's': 40},  # Set transparency and size for scatter plots
        diag_kws={'linewidth': 2.5, 'alpha': 0.9}  # Style the KDE curves
    )

    # Adjust layout and add a title
    plt.suptitle('Comparison of Model Performance Metrics Across Algorithms', y=1.02, fontsize=16)
    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(OUTPUT_FOLDER, 'model_performance_pairplot.png')
    plt.savefig(save_path, dpi=300)
    print(f"Pairplot saved to: {save_path}")
    plt.show()


# ===================================================================
# 3. Main Execution Block
# ===================================================================

def main():
    """
    Main function to execute the data loading and plotting workflow.
    """
    # --- IMPORTANT ---
    # This path is a placeholder. For supplementary materials, it's best to
    # assume the data file is in the same directory as the script.
    performance_data_file = "model_performance_data.csv"

    # Generate the performance comparison plot
    plot_performance_pairplot(performance_data_file)


if __name__ == "__main__":
    main()
