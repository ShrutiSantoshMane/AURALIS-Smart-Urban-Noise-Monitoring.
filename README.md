# AURALIS-Smart-Urban-Noise-Monitoring.
AURALIS is a smart urban noise monitoring prototype using machine learning, audio classification, simulated IoT readings, hotspot visualization, and noise analysis.
# AURALIS – Smart Urban Noise Monitoring System

## Overview
AURALIS is a machine-learning-based prototype for analysing urban sound and exploring noise-monitoring patterns across different locations. It combines audio classification, simulated monitoring data, hotspot visualisation, and model-based analysis in a single project.

## Objectives
- Analyse environmental audio using machine learning.
- Classify sound categories using extracted audio features.
- Explore simulated noise readings across urban locations.
- Visualise potential noise hotspots.
- Support understanding of urban noise patterns through data analysis.

## Features
- **Audio Intelligence:** Extracts audio features such as MFCCs and spectral characteristics for sound classification.
- **Machine Learning:** Uses a Random Forest classifier for audio analysis.
- **Simulated Monitoring:** Displays example readings for multiple locations.
- **Hotspot Visualisation:** Helps explore locations that may need further investigation.
- **Interactive Dashboard:** Presents project outputs through a Streamlit interface.

## Technologies Used
- Python
- Streamlit
- Pandas and NumPy
- Librosa
- Scikit-learn
- Matplotlib and Plotly
- SHAP, where implemented in the project

## Dataset
The audio classification component uses the UrbanSound8K dataset. Please obtain the dataset from its official source and follow its usage terms. Large audio files are not included in this repository.

## Model Results
The Random Forest model achieved approximately **94.17% test accuracy** in the recorded experiment. Results depend on the selected data, features, and evaluation split.

## Important Limitations
This is an academic prototype. The location readings and monitoring scenarios are simulated, not measurements from deployed physical sensors. The displayed values should not be interpreted as calibrated sound-pressure-level measurements. Actual deployment would require suitable sensors, field validation, and calibration.

## Run the Project
1. Install Python.
2. Clone or download this repository.
3. Install the required libraries:

   `pip install -r requirements.txt`

4. Start the dashboard:

   `streamlit run app.py`

Ensure that the required dataset and model files are available in the locations expected by the code.

## Future Scope
- Integration with physical noise sensors.
- Real-time monitoring and alert generation.
- Improved field validation and calibration.
- Additional sound-classification experiments.
- Expanded geographic visualisation.

## Project Type
Academic machine-learning and smart-city prototype.
