# Auto-generated extraction script for EXAMPLE_DATASET

extract_discovered_features <- function(data_file) {
  data <- read.csv(data_file)

  # AUTONOMIC features
  autonomic_vars <- c("HEART_RATE_BASELINE")
  autonomic_data <- data[, autonomic_vars]

  # CIRCADIAN features
  circadian_vars <- c("CORTISOL_PM", "CORTISOL_AM", "MERCURY_HAIR")
  circadian_data <- data[, circadian_vars]

  # SENSORY features
  sensory_vars <- c("SENSORY_PROFILE_TACTILE")
  sensory_data <- data[, sensory_vars]

  # INTEROCEPTION features
  interoception_vars <- c("BP_SYSTOLIC", "SENSORY_PROFILE_TACTILE", "ABR_THRESHOLD")
  interoception_data <- data[, interoception_vars]

  # AUDITORY_PROCESSING features
  auditory_processing_vars <- c("ABR_THRESHOLD")
  auditory_processing_data <- data[, auditory_processing_vars]

  # ENVIRONMENTAL_EXPOSURE features
  environmental_exposure_vars <- c("LEAD_BLOOD", "MERCURY_HAIR")
  environmental_exposure_data <- data[, environmental_exposure_vars]

  # TRACE_MINERALS features
  trace_minerals_vars <- c("ZINC_SERUM", "HEART_RATE_BASELINE", "SENSORY_PROFILE_TACTILE", "MERCURY_HAIR", "VISUAL_ACUITY")
  trace_minerals_data <- data[, trace_minerals_vars]

  return(list(
    autonomic = autonomic_data,
    circadian = circadian_data,
    sensory = sensory_data,
    interoception = interoception_data,
    auditory_processing = auditory_processing_data,
    visual_processing = visual_processing_data,
    environmental_exposure = environmental_exposure_data,
    trace_minerals = trace_minerals_data,
    inflammatory_markers = inflammatory_markers_data,
    metabolic_calculated = metabolic_calculated_data
  ))
}
