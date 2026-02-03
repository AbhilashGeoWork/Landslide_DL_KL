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
  <em>Figure 1. Location and physiography of Kerala, India. Reddish colours represent the Western Ghats region.</em>
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
  <img src="Figures/BI_Formula.png" width="300"/>
  <br>
  <em>Figure 2. Equation for Brightness Index (BI), where R, G, and B stands for Red, Green, and Blue bands respectively.</em>
</p>

<p align="center">
  <img src="Figures/NDVI_Formula.png" width="300"/>
  <br>
  <em>Figure 3. Equation for Normalized Difference Vegetation Index (NDVI), where NIR and R stands for Near-Infrared and Red bands respectively.</em>
</p>

<p align="center">
  <img src="Figures/NDWI_Formula.png" width="300"/>
  <br>
  <em>Figure 4. Equation for Normalized Difference Water Index (NDWI), where G and NIR stands for Green and Near-Infrared bands respectively.</em>
</p>

### Data Preprocessing
- 700 samples were digitized from the study area (350 landslide and 350 non-landslide samples).
- They were divided into 400 samples for training (200 landslide & 200 non-landslide), 200 samples for testing (100 each) and 100 for validation (50 each).
- For each sample, the Planetscope layers (3 meter) were tiled to 128x128 patches (2<sup>7</sup>), covering an area of 384 meter in length and width (128 × 3). The Sentinel-1 SAR layers (10 meter) covered the same spatial extent with 38x38 tiles (≈384 ÷ 10) while ALOS DEM layers (12.5 meter) covered it with 31x31 tiles (≈384 ÷ 12.5). To ensure dimensional consistency and compatibility across inputs, both DEM and SAR layers were resampled to a standardized dimension of 32x32 (2<sup>5</sup>) using the nearest neighbour interpolation method.
- The values in each layer / band was normalized to the range of -1 to 1, with -9 being the placeholder for pixels with no data (Nan, 0, -9999). Since NDVI, aspect-sin, and aspect-cos are already in the range of -1 to 1, normalization was not applied to them.

<p align="center">
  <img src="Figures/Sample.png" width="600"/>
  <br>
  <em>Figure 5. A landslide sample used in the study. Pre-event and post-event true-color (RGB) composites are used only for visualization purposes and are not included as inputs for the segmentation models.</em>
</p>

### Data Availability
Due to size constraints, the full dataset is not included in this repository.
A small sample dataset is provided under `Data`.

## Models
For this study, 5 variants of the U-Net algorithm were used:
- U-Net
- U-Net++
- U-Net++ with ECA (Efficient Channel Attention)
- U-Net++ with ECA and Deep Supervision
- U-Net++ with ECA, Deep Supervision, and ASPP (Atrous Spatial Pyramid Pooling).

For convenience, these models are named Model 1, 2, 3, 4, and 5 respectively in this study.

Since the dataset consists of layers derived from three different sources, each with distinct spatial resolutions and information, three separate encoders are employed in all models, with each encoder processing one set of data layers.

The detailed architecture for these models can be found in `Model_Architecture.py`.