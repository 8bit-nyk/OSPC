# OSPC: Online Sequential Photometric Calibration
Matlab implementation of a sequential photometric calibration algorithm.

## Description
This MATLAB project performs online sequential photometric calibration starting with exposure values to estimate the camera response function (CRF) and vignette mapping of any camera.

## Realted Publication:
**Authors:** Jawad Haidar, Douaa Khalil, Daniel Asmar,

**Title:** OSPC: Online Sequential Photometric Calibration.

**Publication:** Pattern Recognition Letters, Volume 181, 2024, Pages 30-36, ISSN 0167-8655

[**Link to publication**](https://www.sciencedirect.com/science/article/pii/S0167865524000734)

### Citation:
```
@article{haidar2024ospc,
  title={OSPC: Online sequential photometric calibration},
  author={Haidar, Jawad and Khalil, Douaa and Asmar, Daniel},
  journal={Pattern Recognition Letters},
  year={2024},
  publisher={Elsevier}
}
```

## Dependencies
- MATLAB
- Matlab Tollboxes:
  - Computer Vision Toolbox 
  - Curve Fitting Toolbox
  - Image Processing Toolbox
  - Mapping Toolbox
  - Natural-Order Filename Sort 
  - Optimization Toolbox 
  - Statistics and Machine Learning Toolbox 
  - The HDR Toolbox 

## Usage
### Inputs:
The inputs for the algorithm are images along with a text file containing their exposure meta-data.

Sample image folders and exposure files are provided in the **Inputs** directory for quick testing.

To perform OSPC on your own camera simply record a dataset similair to the proivded and feed it to the algorithm as indicated in the coming steps.

### Camera Response Function (CRF) Estimation

To estimate the CRF, use the `ospc_CRF` function. This function takes two arguments: the path to a folder containing images and the path to a file containing corresponding exposure times.

Example usage:

```matlab
image_folder_path = 'path/to/image/folder';
exposure_file_path = 'path/to/exposure/times.txt';
output = ospc_CRF(image_folder_path, exposure_file_path);
```
The output is a normalized CRF curve, which can be saved or plotted as desired.

### Vignetting Estimation

To estimate vignetting, use the estimate_vignette_refactored function. This function requires three arguments: the path to a folder containing images, the path to a file containing corresponding exposure times, and the path to a file containing the CRF data.

Example usage:

```matlab
image_folder_path = 'path/to/image/folder';
exposure_file_path = 'path/to/exposure/times.txt';
crf_file_path = 'path/to/crf/data.txt';
output = estimate_vignette_refactored(image_folder_path, exposure_file_path, crf_file_path);
```
The output is a vignetting curve, which can be saved or plotted as desired.

### Testing
Two test scripts are provided to demonstrate the usage of the main functions:

    test_crf.m: Demonstrates CRF estimation.
    test_vig.m: Demonstrates vignetting estimation.

Run these scripts in MATLAB to test the functionality.

### Outputs:
The outputs of the algorithm are save in an **outputs** directory that gets created automatically.
- The CRF estimation outputs the normalized CRF curve as a text file containing 256 values in ascending order.
- The vignette estimation outputs a 16 bit grayscale vignette map as a png image

## License:
This project is released under [GPLv3 license]

### Contributions:
Contributions to the project are welcome. If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make the necessary changes and commit them.
4. Push your changes to your forked repository.
5. Submit a pull request detailing the changes you have made.
