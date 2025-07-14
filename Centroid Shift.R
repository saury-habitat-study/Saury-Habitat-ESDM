# ===================================================================
# Script to Plot Habitat Centroid Shifts for Pacific Saury
# This script generates a two-panel figure showing the projected migration
# of the habitat centroid from the current period to the 2050s and 2100s
# under different Shared Socioeconomic Pathways (SSPs).
# ===================================================================

# 1. Load Necessary Libraries
# -------------------------------------------------------------------
library(ggplot2)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)
library(dplyr)
library(cowplot) # Used to combine the two plots

# ===================================================================
# 2. Define Core Data
# The centroid coordinates and label positions are hardcoded for reproducibility.
# ===================================================================

# --- Main centroid data ---
centroid_data <- data.frame(
  Scenario = c("Current", "2050s (SSP1-2.6)", "2050s (SSP2-4.5)", "2050s (SSP3-7.0)", "2050s (SSP5-8.5)", 
               "2100s (SSP1-2.6)", "2100s (SSP2-4.5)", "2100s (SSP3-7.0)", "2100s (SSP5-8.5)"),
  Centroid_Longitude = c(155.78, 157.13, 157.32, 157.88, 157.64,
                         157.98, 158.42, 158.82, 159.13),
  Centroid_Latitude = c(41.45, 43.09, 43.22, 43.35, 43.42,
                        43.28, 44.41, 44.21, 44.53)
)

# --- Nudge values for positioning labels to avoid overlap ---
nudge_values <- data.frame(
  Scenario = c("Current", "2050s (SSP1-2.6)", "2050s (SSP2-4.5)", "2050s (SSP3-7.0)", "2050s (SSP5-8.5)", 
               "2100s (SSP1-2.6)", "2100s (SSP2-4.5)", "2100s (SSP3-7.0)", "2100s (SSP5-8.5)"),
  nudge_x = c(0.8, -1.1, -1.2, 1.2, 0, -1.1, -0.4, 1.3, 0.6),
  nudge_y = c(-0.1, -0.1, 0.3, 0, 0.5, -0.3, 0.4, -0.1, 0.3)
)

# ===================================================================
# 3. Data Preparation
# Prepare the data frame for plotting by adding helper columns.
# ===================================================================
cat("Preparing data for plotting...\n")
centroid_data <- left_join(centroid_data, nudge_values, by = "Scenario")
centroid_data$label_lon <- centroid_data$Centroid_Longitude + centroid_data$nudge_x
centroid_data$label_lat <- centroid_data$Centroid_Latitude + centroid_data$nudge_y

# Create columns for 'Period' and 'SSP' to use for grouping and coloring
centroid_data$Period <- ifelse(grepl("2050s", centroid_data$Scenario), "2050s", 
                               ifelse(grepl("2100s", centroid_data$Scenario), "2100s", "Current"))
centroid_data$SSP <- gsub(".*\\((SSP.*)\\).*", "\\1", centroid_data$Scenario)
centroid_data$SSP[centroid_data$Period == "Current"] <- "Current"

# Get world map data and define plot limits
world_map <- ne_countries(scale = "medium", returnclass = "sf")
current_coords <- subset(centroid_data, Period == "Current")
x_limits <- c(155, 161.5)
y_limits <- c(41, 45)

# ===================================================================
# 4. Create the Plots
# Generate two separate plots: one for the 2050s and one for the 2100s.
# ===================================================================

# --- Plot 1: 2050s Centroid Shift ---
cat("Generating plot for the 2050s...\n")
data_2050 <- subset(centroid_data, Period %in% c("Current", "2050s"))
p_2050 <- ggplot(data = data_2050) +
  geom_sf(data = world_map, fill = "grey85", color = "white") +
  geom_segment(data = subset(data_2050, Period == "2050s"),
               aes(x = current_coords$Centroid_Longitude, y = current_coords$Centroid_Latitude, 
                   xend = Centroid_Longitude, yend = Centroid_Latitude),
               arrow = arrow(length = unit(0.2, "cm")), color = "blue", linewidth = 0.8) +
  geom_point(aes(x = Centroid_Longitude, y = Centroid_Latitude, shape = Period, fill = Period), size = 4, color = "black") +
  geom_segment(aes(x = Centroid_Longitude, y = Centroid_Latitude, xend = label_lon, yend = label_lat), color = "grey50", linewidth = 0.5) +
  geom_label(aes(x = label_lon, y = label_lat, label = SSP), size = 4) +
  scale_shape_manual(values = c("Current" = 23, "2050s" = 21)) +
  scale_fill_manual(values = c("Current" = "yellow", "2050s" = "blue")) +
  annotate("text", x = x_limits[1] + 0.2, y = y_limits[2] - 0.2, label = "2050s", hjust = 0, vjust = 1, size = 8, fontface = "bold") +
  annotate("text", x = x_limits[2] - 0.1, y = y_limits[1] + 0.1, label = "(a)", hjust = 1, vjust = 0, size = 10, fontface = "bold") +
  coord_sf(xlim = x_limits, ylim = y_limits, expand = FALSE) +
  theme_bw() +
  labs(x = NULL, y = NULL) +
  theme(legend.position = "none", panel.grid = element_blank(), 
        axis.text = element_text(size = 12, face = "bold"))

# --- Plot 2: 2100s Centroid Shift ---
cat("Generating plot for the 2100s...\n")
data_2100 <- subset(centroid_data, Period %in% c("Current", "2100s"))
p_2100 <- ggplot(data = data_2100) +
  geom_sf(data = world_map, fill = "grey85", color = "white") +
  geom_segment(data = subset(data_2100, Period == "2100s"),
               aes(x = current_coords$Centroid_Longitude, y = current_coords$Centroid_Latitude, 
                   xend = Centroid_Longitude, yend = Centroid_Latitude),
               arrow = arrow(length = unit(0.2, "cm")), color = "red", linewidth = 0.8) +
  geom_point(aes(x = Centroid_Longitude, y = Centroid_Latitude, shape = Period, fill = Period), size = 4, color = "black") +
  geom_segment(aes(x = Centroid_Longitude, y = Centroid_Latitude, xend = label_lon, yend = label_lat), color = "grey50", linewidth = 0.5) +
  geom_label(aes(x = label_lon, y = label_lat, label = SSP), size = 4) +
  scale_shape_manual(values = c("Current" = 23, "2100s" = 22)) +
  scale_fill_manual(values = c("Current" = "yellow", "2100s" = "red")) +
  annotate("text", x = x_limits[1] + 0.2, y = y_limits[2] - 0.2, label = "2100s", hjust = 0, vjust = 1, size = 8, fontface = "bold") +
  annotate("text", x = x_limits[2] - 0.1, y = y_limits[1] + 0.1, label = "(b)", hjust = 1, vjust = 0, size = 10, fontface = "bold") +
  coord_sf(xlim = x_limits, ylim = y_limits, expand = FALSE) +
  theme_bw() +
  labs(x = NULL, y = NULL) +
  theme(legend.position = "none", panel.grid = element_blank(),
        axis.text = element_text(size = 12, face = "bold"))

# ===================================================================
# 5. Combine and Save the Final Figure
# ===================================================================
cat("Combining plots into a single figure...\n")
final_plot <- plot_grid(p_2050, p_2100, ncol = 2)

# Save the combined plot to a file
ggsave("Figure_Centroid_Shift.png", final_plot, width = 16, height = 8, dpi = 300)

cat("Process complete. Final plot saved as 'Figure_Centroid_Shift.png'.\n")
print(final_plot)
