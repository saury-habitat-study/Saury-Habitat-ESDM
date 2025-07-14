# Code for "Habitat Shifts of Pacific Saury (Cololabis saira) population in the High Seas of the North Pacific under medium to long-term Climate Scenarios Based on Vessel Position Da-ta and Ensemble Species Distribution Models"

This repository contains the R and Python scripts used for the analysis in the manuscript titled "Habitat Shifts of Pacific Saury (*Cololabis saira*) population in the High Seas of the North Pacific under medium to long-term Climate Scenarios...".

## Code Structure and Acknowledgment

This research utilizes a combination of established open-source tools and custom-developed models. The code is organized as follows:

### 1. Ensemble Species Distribution Modeling (ESDM)

The core ensemble species distribution modeling framework is implemented using the **`biomod2` R package**. We gratefully acknowledge the developers of this powerful open-source tool. The scripts related to running `biomod2`, such as `habitat_transition_analysis.R`, are primarily workflows that apply this package to our specific dataset.

-   **`biomod2` Official Source:** [https://github.com/biomodhub/biomod2]

### 2. Custom Models and Scripts (Original Contributions)

The following components were developed independently by the authors for this study:

-   **Deep Learning Model for Fishing Behavior Identification :** The Python script defining the CNN-LSTM architecture and the data preprocessing pipeline for identifying fishing states from AIS data is our original work.
-   **Threshold-based Classification Script :** The R script for classifying vessel states using a threshold-based method and generating the corresponding ridge plots is our original work.
-   **Data Visualization Scripts:** All scripts used to generate the final figures for the manuscript (e.g., `plot_VIF.py`, `plot_factor_contributions.py`, `plot_centroid_shift.R`) are custom scripts developed for this study.

## How to Use

Please refer to the comments within each script for details on execution. The necessary data to run these scripts can be made available from the corresponding author upon reasonable request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
