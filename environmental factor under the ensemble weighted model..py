import pandas as pd
import matplotlib.pyplot as plt
import os

# ===================================================================
# 1. Configuration and Constants
# This section defines all constants for plotting to ensure consistency.
# ===================================================================

# Define the output folder for plots
OUTPUT_FOLDER = "response_curve_plots"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Define the desired order for plotting the factors
# This ensures the subplots always appear in a consistent sequence.
DESIRED_FACTOR_ORDER = ['thetao', 'chl', 'mld', 'cv', 'so']

# Define labels and units for each environmental factor for clear plotting.
# Uses LaTeX formatting for special characters.
FACTOR_LABELS = {
    'thetao': {'xlabel': 'SST (°C)', 'ylabel': 'Habitat Suitability'},
    'chl': {'xlabel': 'CHL (mg·m$^{-3}$)', 'ylabel': 'Habitat Suitability'},
    'mld': {'xlabel': 'MLD (m)', 'ylabel': 'Habitat Suitability'},
    'cv': {'xlabel': 'CV (m·s$^{-1}$)', 'ylabel': 'Habitat Suitability'},
    'so': {'xlabel': 'SSS (‰)', 'ylabel': 'Habitat Suitability'}
}


# ===================================================================
# 2. Core Functions
# These functions handle data loading and visualization.
# ===================================================================

def load_response_data(file_path):
    """
    Loads response curve data from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def plot_all_response_curves(df):
    """
    Generates and saves a multi-panel plot showing the response curve
    for each environmental factor.
    """
    if df is None or df.empty:
        print("No data available to plot.")
        return

    # Create a 3x2 grid for the subplots
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for i, factor in enumerate(DESIRED_FACTOR_ORDER):
        # Filter the dataframe for the current factor and sort by value
        subset = df[df['expl.name'] == factor].sort_values('expl.val')

        if subset.empty:
            print(f"Warning: No data found for factor '{factor}'. Skipping plot.")
            continue

        ax = axes[i]

        # Plot the response curve
        ax.plot(subset['expl.val'], subset['pred.val'], linewidth=2.5, color='#003366')

        # Set labels using the predefined dictionary
        labels = FACTOR_LABELS.get(factor, {})
        ax.set_xlabel(labels.get('xlabel', factor), fontsize=14)
        ax.set_ylabel(labels.get('ylabel', 'Probability'), fontsize=14)

        # Configure plot aesthetics for a clean, professional look
        ax.grid(False)  # Remove grid lines
        ax.set_facecolor('white')  # Set background to white
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.2)

        ax.tick_params(axis='both', which='major', labelsize=12)

    # Hide any unused subplots in the grid
    for j in range(len(DESIRED_FACTOR_ORDER), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(pad=3.0)

    # Save the final figure
    save_path = os.path.join(OUTPUT_FOLDER, 'all_environmental_response_curves.png')
    plt.savefig(save_path, dpi=300)
    print(f"Multi-panel response curve plot saved to: {save_path}")
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
    data_file_path = "response_curves_data.csv"

    # Load the data
    response_data = load_response_data(data_file_path)

    # Generate the plot
    plot_all_response_curves(response_data)


if __name__ == "__main__":
    main()
