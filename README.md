# Landslide Mapping in Kerala From Multi-Sensor Satellite Data Using Deep Learning.

## Overview
Landslide detection and mapping are crucial for hazard assessment and disaster management.
This project implements deep learning–based segmentation models to map landslides from multi-sensor satellite data.

## Objectives
- Prepare a landslide dataset from Kerala involving data from multiple satellite sensors.
- Implement deep learning models for mapping landslides from this dataset.
- Compare model performance using visual comparison and evaluation metrics. 

## Study Area
Study area covers the landslide-prone Western Ghats region in Kerala, India characterized by dissected hills, valleys, and plateau along with monsoon-driven rainfall.
<p align="center">
  <img src="Figures/KeyMap.jpg" width="600"/>
  <br>
  <em>Figure 1. Location and physiography of Kerala, India.</em>
</p>

## Dataset

### Sources
- PlanetScope 3 meter multi-spectral imagery: post-event and pre-event.
- Sentinel-1 10 meter SAR (Synthetic Aperture Radar) imagery: post-event and pre-event.
- ALOS DEM 12.5 meter layers.

### Features
- PlanetScope: NIR (Near-Infrared), BI (Brightness Index), NDVI (Normalized Difference Vegetation Index), NDWI (Normalized Difference Water Index).  
- Sentinel-1: VV, VH, VV-VH (band-wise difference)
- ALOS DEM (Digital Elevation Model): Elevation, Slope, Aspect, TWI (Topographic Wetness Index)

<p align="center">
  <img src="Figures/Sample.png" width="600"/>
  <br>
  <em>Figure 2. A landslide sample used in the study. Pre-event and post-event true-color (RGB) composites are used only for visualization purposes and are not included as inputs for the segmentation models. All layers are normalized to the range -1 to 1, with -9 as a placeholder for NoData.</em>
</p>

### Data Availability
Due to size constraints, the full dataset is not included in this repository.
A small sample dataset is provided under `Data`.
