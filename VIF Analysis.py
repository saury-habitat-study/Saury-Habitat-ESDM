import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ===================================================================
# 1. Configuration and Constants
# ===================================================================

# Define the output folder for the plot
OUTPUT_FOLDER = "vif_analysis_plots"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Define a color palette for the plot
# Using a single color or a simple palette is often cleaner for this type of plot.
BAR_COLOR = '#3498db'  # A nice, clean blue


# ===================================================================
# 2. Core Plotting Function
# ===================================================================

def plot_vif_analysis(data_path):
    """
    Loads VIF scores from a CSV file and generates a horizontal bar chart
    to visualize the collinearity among environmental variables.

    Args:
        data_path (str): The path to the CSV file containing VIF data.
                         The CSV should have 'Variable' and 'VIF' columns.
    """
    try:
        vif_df = pd.read_csv(data_path)
        print(f"VIF data loaded successfully. Shape: {vif_df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Check for required columns
    if not all(col in vif_df.columns for col in ['Variable', 'VIF']):
        print("Error: CSV file must contain 'Variable' and 'VIF' columns.")
        return

    # Sort data by VIF score for better visualization
    vif_df = vif_df.sort_values(by='VIF', ascending=True)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the horizontal bar plot
    sns.barplot(x='VIF', y='Variable', data=vif_df, color=BAR_COLOR, orient='h', ax=ax)

    # Add the VIF=5 threshold line for reference
    ax.axvline(5, color='darkorange', linestyle='--', linewidth=2)
    ax.text(5.1, 0.5, 'Moderate Collinearity Threshold (VIF=5)',
            color='darkorange', fontsize=10, transform=ax.get_yaxis_transform())

    # Add data labels inside the bars for clarity
    for index, row in vif_df.iterrows():
        # Position the text slightly to the left of the bar's end
        text_position = row.VIF - 0.15
        ax.text(
            x=text_position,
            y=index,
            s=f'{row.VIF:.2f}',
            color='white',
            ha="right",
            va="center",
            fontweight='bold'
        )

    # Configure plot aesthetics
    ax.grid(False)  # Remove background grid lines
    ax.set_xlabel('Variance Inflation Factor (VIF) Score', fontsize=12)
    ax.set_ylabel('Environmental Variable', fontsize=12)
    ax.set_xlim(0, 10)  # Set x-axis limit for better focus

    # Save and display the plot
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_FOLDER, 'vif_collinearity_analysis.png')
    plt.savefig(save_path, dpi=300)
    print(f"VIF analysis plot saved to: {save_path}")
    plt.show()


# ===================================================================
# 3. Main Execution Block
# ===================================================================

def main():
    """
    Main function to execute the plotting workflow.
    """
    # --- IMPORTANT ---
    # This path is a placeholder. For supplementary materials, it's best to
    # assume the data file is in the same directory as the script.
    # The CSV should contain two columns: 'Variable' and 'VIF'.
    vif_data_file = "vif_scores.csv"

    # Create a dummy CSV file for demonstration if it doesn't exist
    if not os.path.exists(vif_data_file):
        print(f"Creating a dummy data file: {vif_data_file}")
        dummy_data = {
            'Variable': ['CHL', 'SSS', 'SST', 'MLD', 'CV'],
            'VIF': [1.71, 3.51, 4.75, 1.64, 2.38]
        }
        pd.DataFrame(dummy_data).to_csv(vif_data_file, index=False)

    # Generate the VIF analysis plot
    plot_vif_analysis(vif_data_file)


if __name__ == "__main__":
    main()
