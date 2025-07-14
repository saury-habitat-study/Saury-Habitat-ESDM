import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# ===================================================================
# 1. Configuration and Constants
# This section defines all constants for plotting to ensure consistency.
# ===================================================================

# Define the output folder for plots
OUTPUT_FOLDER = "contribution_plots"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Map internal factor names to display names
FACTOR_MAPPING = {
    'thetao': 'SST',
    'chl': 'CHL',
    'mld': 'MLD',
    'cv': 'CV',
    'so': 'SSS'
}

# Define the desired display order for factors in the final plot
DESIRED_FACTOR_ORDER = ['SST', 'CHL', 'MLD', 'CV', 'SSS']

# Define colors for each factor to match the manuscript's figures
FACTOR_COLORS = {
    'thetao': '#66ccff',  # SST (Light Blue)
    'chl': '#003366',  # CHL (Dark Blue)
    'mld': '#3399ff',  # MLD (Blue)
    'cv': '#0066cc',  # CV (Medium Blue)
    'so': '#ff9900'  # SSS (Orange)
}


# ===================================================================
# 2. Core Functions
# These functions handle data loading, processing, and visualization.
# ===================================================================

def load_and_process_data(file_path):
    """
    Loads contribution data from a CSV, standardizes factor names,
    and calculates relative contribution percentages.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")

        # Standardize environmental factor names for internal processing
        df['expl.var'] = df['expl.var'].replace({v: k for k, v in FACTOR_MAPPING.items()})

        # Filter out any factors not defined in our color map
        df = df[df['expl.var'].isin(FACTOR_COLORS.keys())]

        # Calculate relative contribution percentage for each factor within each model run
        df['contribution_pct'] = df.groupby(['algo'])['var.imp'].transform(
            lambda x: x / x.sum() * 100
        )
        return df
    except Exception as e:
        print(f"Error loading or processing data: {e}")
        return None


def plot_model_comparison(df):
    """
    Generates and saves a grouped bar chart comparing factor contributions
    across all models, ordered and colored as specified.
    """
    if df is None or df.empty:
        print("No data available to plot.")
        return

    # Calculate the mean contribution for each factor in each model
    summary_df = df.groupby(['algo', 'expl.var'])['contribution_pct'].mean().reset_index()

    # Map internal names to display names for plotting
    summary_df['Environmental Factor'] = summary_df['expl.var'].map(FACTOR_MAPPING)

    # Set the factor order for the plot legend and colors
    summary_df['Environmental Factor'] = pd.Categorical(
        summary_df['Environmental Factor'],
        categories=DESIRED_FACTOR_ORDER,
        ordered=True
    )

    # Prepare the color palette in the correct order
    palette = {factor: FACTOR_COLORS[key] for key, factor in FACTOR_MAPPING.items()}

    # Create the plot
    plt.figure(figsize=(14, 8))
    sns.barplot(
        x='algo',
        y='contribution_pct',
        hue='Environmental Factor',
        data=summary_df,
        palette=palette,
        hue_order=DESIRED_FACTOR_ORDER,  # Ensure legend order is correct
        edgecolor='black',
        linewidth=0.7
    )

    # Configure plot aesthetics
    plt.xlabel('Models', fontsize=14)
    plt.ylabel('Average Contribution (%)', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 60)  # Set y-axis limit to 60%
    plt.grid(False)  # Remove grid lines
    plt.legend(title='Environmental Factors', loc='upper right', fontsize=12)
    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(OUTPUT_FOLDER, 'all_models_contribution_comparison.png')
    plt.savefig(save_path, dpi=300)
    print(f"Comparison plot saved to: {save_path}")
    plt.show()


# ===================================================================
# 3. Main Execution Block
# ===================================================================

def main():
    """
    Main function to execute the data loading, processing, and plotting workflow.
    """
    # --- IMPORTANT ---
    # Replace this placeholder with the path to your data file.
    # For supplementary materials, it's best to assume the data file
    # is in the same directory as the script.
    data_file_path = "c-models.csv"

    # Load and process the data
    contribution_data = load_and_process_data(data_file_path)

    # Generate the comparison plot
    plot_model_comparison(contribution_data)


if __name__ == "__main__":
    main()
