# Label conversions
BACKGROUND = 0
SANDEEL = 1
OTHER = 2

# Ignore label values
LABEL_IGNORE_VAL = -100
LABEL_BOUNDARY_VAL = -100  # If crop is outside data, set all values outside data array to this value in labels
LABEL_OVERLAP_VAL = -70  # When predicting on overlapping crops, set overlap area to this value in labels
LABEL_SEABED_MASK_VAL = -50  # Set all label values beneath seabed to this value
LABEL_REFINE_BOUNDARY_VAL = -30  # Set all label values with low frequency response to this value
LABEL_UNUSED_SPECIES = -10  # Set label for species other than sandeel to this value

# Data boundary val -> to fill missing values BEFORE decibel transform
DATA_BOUNDARY_VAL = 0

