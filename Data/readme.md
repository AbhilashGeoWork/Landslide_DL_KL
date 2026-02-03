This folder contains a few samples of the data prepared for this study in numpy array format (.npy). All channels in the arrays, except `MASK_SAMPLE.npy`, are individually normalized to the range -1 to 1 with -9 as NoData placeholder.
## File Naming
- MASK: Binary mask showing landslide in the input.
- MSS: Multi-spectral layers from Planetscope for segmentation.
- RGBI: Blue, Greed, Red, and NIR layers from Planetscope for visualization.
- DEM: Terrain layers derived from ALOS DEM.
- SAR: Sentinel-1 SAR layers.

## Array Dimensions
The shape of these arrays are in the format:
No. of samples x XY dimension x no. of bands / channels
- MASK: 10x128x128x1
- MSS: 10x128x128x8
- RGBI: 10x128x128x8
- DEM: 10x32x32x5
- SAR: 10x32x32x6

<p align="center"><b>Table 1. Layers in MSS_SAMPLE by index.</b></p>

<table align="center">
  <tr>
    <th>Index</th>
    <th>Layer</th>
  </tr>
  <tr>
    <td>1</td>
    <td>NIR (Post-event)</td>
  </tr>
  <tr>
    <td>2</td>
    <td>BI (Post-event)</td>
  </tr>
  <tr>
    <td>3</td>
    <td>NDVI (Post-event)</td>
  </tr>
  <tr>
    <td>4</td>
    <td>NDWI (Post-event)</td>
  </tr>
  <tr>
    <td>5</td>
    <td>NIR (Pre-event)</td>
  </tr>
  <tr>
    <td>6</td>
    <td>BI (Pre-event)</td>
  </tr>
  <tr>
    <td>7</td>
    <td>NDVI (Pre-event)</td>
  </tr>
  <tr>
    <td>8</td>
    <td>NDWI (Pre-event)</td>
  </tr>  
</table>

<p align="center"><b>Table 2. Layers in RGBI_SAMPLE by index.</b></p>

<table align="center">
  <tr>
    <th>Index</th>
    <th>Layer</th>
  </tr>
  <tr>
    <td>1</td>
    <td>Blue (Post-event)</td>
  </tr>
  <tr>
    <td>2</td>
    <td>Green (Post-event)</td>
  </tr>
  <tr>
    <td>3</td>
    <td>Red (Post-event)</td>
  </tr>
  <tr>
    <td>4</td>
    <td>NIR (Post-event)</td>
  </tr>
  <tr>
    <td>5</td>
    <td>Blue (Pre-event)</td>
  </tr>
  <tr>
    <td>6</td>
    <td>Green (Pre-event)</td>
  </tr>
  <tr>
    <td>7</td>
    <td>Red (Pre-event)</td>
  </tr>
  <tr>
    <td>8</td>
    <td>NIR (Pre-event)</td>
  </tr>  
</table>

<p align="center"><b>Table 3. Layers in DEM_SAMPLE by index.</b></p>

<table align="center">
  <tr>
    <th>Index</th>
    <th>Layer</th>
  </tr>
  <tr>
    <td>1</td>
    <td>Elevation</td>
  </tr>
  <tr>
    <td>2</td>
    <td>Slope</td>
  </tr>
  <tr>
    <td>3</td>
    <td>Aspect - Sine</td>
  </tr>
  <tr>
    <td>4</td>
    <td>Aspect - Cosine</td>
  </tr>
  <tr>
    <td>5</td>
    <td>TWI</td>
  </tr>
  </table>

<p align="center"><b>Table 4. Layers in SAR_SAMPLE by index.</b></p>

<table align="center">
  <tr>
    <th>Index</th>
    <th>Layer</th>
  </tr>
  <tr>
    <td>1</td>
    <td>VV (Post-event)</td>
  </tr>
  <tr>
    <td>2</td>
    <td>VH (Post-event)</td>
  </tr>
  <tr>
    <td>3</td>
    <td>VV-VH (Post-event)</td>
  </tr>
  <tr>
    <td>4</td>
    <td>VV (Pre-event)</td>
  </tr>
  <tr>
    <td>5</td>
    <td>VH (Pre-event)</td>
  </tr>
  <tr>
    <td>6</td>
    <td>VV-VH (Pre-event)</td>
  </tr>
  </table>