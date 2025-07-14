# ===================================================================
# Script to Analyze and Plot Habitat Transition Under Future Scenarios
# 
# This script performs the following steps:
# 1. Loads current and future habitat suitability raster files (.asc).
# 2. Reclassifies habitat suitability into four categories.
# 3. Calculates a transition map showing areas of habitat improvement, 
#    degradation, loss, and stability.
# 4. Generates a multi-panel plot visualizing these transitions for
#    multiple climate scenarios.
# 5. Calculates and outputs the area for each transition type.
# ===================================================================

# 1. Load Necessary Libraries
# -------------------------------------------------------------------
library(raster)
library(ggplot2)
library(dplyr)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)

# ===================================================================
# 2. Configuration and Setup
# Users should modify the paths in this section.
# ===================================================================

# --- Define Input and Output Paths (Use placeholders) ---
# It's assumed that all .asc files are located in this folder.
input_folder <- "path/to/your/asc_files" 
output_folder <- "path/to/your/output_plots"

# --- Define Key File Names and GIS Data Paths (Use placeholders) ---
current_file_name <- "current_habitat.asc"
shp_file_path <- "path/to/your/shapefile_folder"
shp_layer_name <- "your_shapefile_layer"

# --- Create output directory if it doesn't exist ---
if (!dir.exists(output_folder)) {
  dir.create(output_folder, recursive = TRUE)
}

# ===================================================================
# 3. Core Functions
# ===================================================================

#' Reclassify a continuous habitat suitability raster into discrete categories.
#' 
#' @param r A raster object with suitability values from 0 to 1.
#' @return A reclassified raster with integer values representing categories.
reclassify_habitat <- function(r) {
  # Define reclassification matrix: [from, to, becomes]
  m <- c(0, 0.4, 1,    # 0-0.4 -> Unsuitable (1)
         0.4, 0.6, 2,  # 0.4-0.6 -> Low Suitability (2)
         0.6, 0.8, 3,  # 0.6-0.8 -> Moderate Suitability (3)
         0.8, 1.0, 4)  # 0.8-1.0 -> High Suitability (4)
  rclmat <- matrix(m, ncol=3, byrow=TRUE)
  return(reclassify(r, rclmat))
}

#' Create a habitat transition raster from current and future classified rasters.
#' 
#' @param current_classified The reclassified raster for the current period.
#' @param future_classified The reclassified raster for a future period.
#' @return A transition raster with codes: 1=Improvement, 2=Degradation, 3=Loss, 4=Stable Suitable.
create_transition_raster <- function(current_classified, future_classified) {
  transition_raster <- raster(current_classified) # Create a template
  transition_raster[current_classified < future_classified] <- 1  # Improvement
  transition_raster[current_classified > future_classified & future_classified > 1] <- 2  # Degradation
  transition_raster[current_classified > 1 & future_classified == 1] <- 3  # Loss
  transition_raster[current_classified > 1 & current_classified == future_classified] <- 4  # Stable Suitable
  return(transition_raster)
}

# ===================================================================
# 4. Main Data Processing Workflow
# This section would typically loop through all future files.
# For this supplementary script, we demonstrate the logic with placeholders.
# ===================================================================

# --- Define plot aesthetics ---
transition_colors <- c("Improvement" = "#3B9AB2", "Degradation" = "#E1AF00", 
                       "Loss" = "#9400D3", "Stable Suitable" = "#F21A00")
transition_labels <- c("Improvement", "Degradation", "Loss", "Stable Suitable")

# --- Example of processing one file (the full script would loop this) ---
# In the full script, a loop would iterate through each future scenario file.
# Here, we show the logic for a single generic future file.

# 1. Load and prepare GIS base layers
# world_map <- ne_countries(scale = "medium", returnclass = "sf")
# eez_shape <- st_read(dsn = shp_file_path, layer = shp_layer_name)
# crop_extent <- extent(135, 180, 30, 65)

# 2. Load and process current raster
# current_raster_raw <- raster(file.path(input_folder, current_file_name))
# current_raster_cropped <- crop(current_raster_raw, crop_extent)
# current_raster_masked <- mask(current_raster_cropped, eez_shape)
# current_classified <- reclassify_habitat(current_raster_masked)

# 3. Process a future raster (example)
# future_file_path <- file.path(input_folder, "future_scenario_example.asc")
# future_raster_raw <- raster(future_file_path)
# future_raster_resampled <- resample(future_raster_raw, current_raster_masked, method='ngb')
# future_raster_masked <- mask(future_raster_resampled, eez_shape)
# future_classified <- reclassify_habitat(future_raster_masked)

# 4. Create transition map
# transition_raster <- create_transition_raster(current_classified, future_classified)

# 5. Convert to data frame for plotting
# transition_df <- as.data.frame(transition_raster, xy = TRUE)
# names(transition_df) <- c("x", "y", "transition_code")
# transition_df <- na.omit(transition_df)
# transition_df$Transition <- factor(transition_df$transition_code, levels = 1:4, labels = transition_labels)

# The loop would collect 'transition_df' for all scenarios into a single large data frame.
# The final plot would then be generated using ggplot2 and facet_wrap, as shown below.

# ===================================================================
# 5. Example Plotting Code
# This demonstrates how the final multi-panel plot is generated.
# It assumes a data frame 'all_scenarios_df' has been created from the loop above.
# ===================================================================

# # Fictional data frame for plotting demonstration
# all_scenarios_df <- ... 

# p_combined <- ggplot() +
#   geom_raster(data = all_scenarios_df, aes(x = x, y = y, fill = Transition)) +
#   geom_sf(data = world_map, fill = "black", color = "grey20") +
#   geom_sf(data = eez_shape, fill = NA, color = "black") +
#   scale_fill_manual(values = transition_colors, name = "Habitat Transition") +
#   coord_sf(xlim = c(135, 180), ylim = c(30, 53), expand = FALSE) +
#   facet_wrap(~ Scenario, ncol = 2) +
#   theme_bw() +
#   labs(x = "Longitude", y = "Latitude") +
#   theme(
#     legend.position = "right",
#     strip.text = element_text(size = 12, face = "bold"),
#     axis.text = element_text(size = 10)
#   )

# # Save the plot
# ggsave("Habitat_Transition_Map.png", p_combined, width = 12, height = 20, dpi = 300)

cat("This script outlines the methodology for creating the habitat transition maps.\n")
cat("The core logic involves reclassifying habitat suitability and comparing current vs. future rasters.\n")

