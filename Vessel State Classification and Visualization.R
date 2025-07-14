# ===================================================================
# Script to Classify Vessel Operational States and Visualize Distributions
#
# This script performs the following steps:
# 1. Loads raw vessel data from an Excel file.
# 2. Classifies vessel activity into distinct operational states based on
#    a set of predefined rules (speed, course change, time, etc.).
# 3. Generates a two-panel ridge plot to visualize the distribution of
#    vessel speed and course difference for each classified state.
# ===================================================================

# 1. Load Necessary Libraries
# -------------------------------------------------------------------
# Ensure all required packages are installed.
# install.packages(c("dplyr", "ggplot2", "lubridate", "openxlsx", "ggridges", "patchwork"))

library(dplyr)
library(ggplot2)
library(lubridate)
library(openxlsx)
library(ggridges)
library(patchwork)

# ===================================================================
# 2. Global Configuration (Colors and Order)
# -------------------------------------------------------------------
# Define a consistent color palette for each operational state.
COLOR_PALETTE <- c(
  "Sailing" = "#004488", "Searching" = "#DDAA33", "Slowing" = "#BB5566",
  "Drifting" = "#117733", "Fishing" = "#CC3311", "Shelter_Sailing" = "#6699CC",
  "Shelter_Drifting" = "#EE7733", "Adjusting" = "#999999"
)

# Define the desired order for states on the y-axis.
STATUS_ORDER <- rev(c(
  "Sailing", "Searching", "Slowing", "Fishing", "Drifting", 
  "Adjusting", "Shelter_Sailing", "Shelter_Drifting"
))

# ===================================================================
# 3. Data Loading and Processing
# -------------------------------------------------------------------
# --- IMPORTANT ---
# This path is a placeholder. For supplementary materials, it's best to
# assume the data file is in the same directory as the script.
data_file_path <- "vessel_activity_data.xlsx"

# Load the data
df <- read.xlsx(data_file_path, sheet = 1)

# Process the data: create datetime objects and classify states
df_processed <- df %>%
  mutate(
    datetime = make_datetime(year, month, day, hour, minute),
    hour = hour(datetime)
  ) %>%
  # =================================================================
# --- Vessel State Classification Logic (Details Omitted) ---
# NOTE: In this step, a set of threshold-based rules are applied to classify
# vessel activity into operational states (e.g., "Fishing", "Sailing").
# The classification is based on variables such as speed, Tcourse, hour,
# and height. The specific `case_when` implementation containing the
# exact thresholds is omitted here for brevity, as detailed in the manuscript's
# methods section. The result of this process is the 'Vessel_State' column.
# =================================================================
mutate(Vessel_State = case_when(
  # [Threshold-based classification logic is applied here]
  # This section is intentionally left blank in this supplementary script.
  # For example: speed >= 0.1 & speed < 4 & (hour >= 22 | hour < 6) ~ "Fishing",
  # ... other rules ...
  TRUE ~ "Adjusting" # A default case would exist
)) %>%
  
  # A simple smoothing step
  mutate(Vessel_State = ifelse(
    Vessel_State == "Adjusting" & lag(Vessel_State) == lead(Vessel_State),
    lag(Vessel_State),
    Vessel_State
  )) %>%
  mutate(Vessel_State = factor(Vessel_State, levels = STATUS_ORDER)) %>%
  filter(!is.na(Vessel_State))

# Create y-axis labels that include the sample size for each state
status_counts <- df_processed %>%
  count(Vessel_State, .drop = FALSE)

y_axis_labels <- status_counts %>%
  arrange(factor(Vessel_State, levels = STATUS_ORDER)) %>%
  mutate(label = paste0(Vessel_State, " (n=", n, ")")) %>%
  pull(label)

# ===================================================================
# 4. Generate Individual Subplots
# -------------------------------------------------------------------

# --- Subplot A: Speed Distribution ---
speed_data_for_ridge <- df_processed %>% filter(speed >= 0, speed <= 15)

speed_plot <- ggplot(speed_data_for_ridge, aes(x = speed, y = Vessel_State, fill = Vessel_State)) +
  geom_density_ridges(quantile_lines = TRUE, alpha = 0.7, scale = 0.9, rel_min_height = 0.01, vline_linetype = "dashed") +
  scale_fill_manual(values = COLOR_PALETTE, guide = "none") +  
  scale_x_continuous(name = "Speed (knots)", expand = c(0, 0), limits = c(0, 15)) +
  scale_y_discrete(name = NULL, expand = c(0.01, 0), labels = y_axis_labels) +
  theme_classic(base_size = 14) +
  theme(
    axis.text.y = element_text(size = 12, color = "black"),
    axis.text.x = element_text(size = 12, color = "black"),
    axis.ticks = element_line(color = "black"),
    axis.line = element_line(linewidth = 0.5),
    axis.title.x = element_text(size = 14, color = "black")
  )

# --- Subplot B: Heading Difference Distribution ---
heading_plot <- ggplot(df_processed, aes(x = Tcourse, y = Vessel_State, fill = Vessel_State)) +
  geom_density_ridges(quantile_lines = TRUE, alpha = 0.7, scale = 0.9, rel_min_height = 0.01, vline_linetype = "dashed") +
  scale_fill_manual(values = COLOR_PALETTE, guide = "none") +
  scale_x_continuous(
    name = "Heading Difference (degrees)", limits = c(0, 180),
    breaks = seq(0, 180, by = 45), expand = c(0, 0)
  ) +
  scale_y_discrete(name = NULL, expand = c(0.01, 0)) +
  theme_classic(base_size = 14) +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    axis.text.x = element_text(size = 12, color = "black"),
    axis.line.x = element_line(linewidth = 0.5),
    axis.ticks.x = element_line(color = "black"),
    axis.title.x = element_text(size = 14, color = "black")
  )

# ===================================================================
# 5. Combine Plots and Save
# -------------------------------------------------------------------

# Combine the two plots side-by-side using patchwork
combined_plot <- speed_plot + heading_plot

# Add 'A' and 'B' tags to the subplots
final_plot <- combined_plot +
  plot_annotation(
    tag_levels = 'A',
    theme = theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  ) &
  theme(
    plot.tag = element_text(size = 20, face = "plain")
  )

# Display the final combined plot
print(final_plot)

# Save the plot to a file
ggsave("Vessel_Behavior_Distributions.png", plot = final_plot, width = 11, height = 7, dpi = 300)
