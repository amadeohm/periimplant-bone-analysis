# Classical Computer Vision Pipeline for Peri-Implant Bone-Level Extraction

**Author:** A. Huerta Moncho
**Year:** 2025

## Description

This repository contains a classical computer vision and mathematical morphology pipeline developed to automate the extraction and analysis of peri-implant bone levels from dental periapical radiographs. The pipeline integrates robust image preprocessing, adaptive thresholding, segmentation, and morphological operations to accurately delineate the peri-implant bone contour.

## Key Functionalities

* **Grayscale Conversion:** Converts RGB images to single-channel grayscale images for further processing.
* **Adaptive Local Mean Thresholding:** Applies adaptive thresholding based on local mean intensities to enhance relevant anatomical structures.
* **Median Filtering:** Reduces impulse noise while preserving critical structural edges.
* **Multi-Otsu Segmentation:** Automatically segments images into multiple classes representing different anatomical and implant materials.
* **Morphological Filtering and Hole Filling:** Removes irrelevant small regions and fills holes to ensure coherent segmentation.
* **Conditioned Morphological Gradient:** Extracts precise bone edges by applying conditions based on class transitions and validated edge regions.

## Installation

### Requirements

* Python 3.x
* OpenCV
* NumPy
* scikit-image
* Matplotlib

### Install Dependencies

```bash
pip install opencv-python numpy scikit-image matplotlib
```

## Usage

### Example

```python
import cv2
from pipeline import run_pipeline

params = {
    "log_gain": 1.0,
    "prep_mode": "log",
    "median_ksize": 5,
    "block_size": 15,
    "C": 10,
    "n_classes": 3,
    "min_area": 500,
    "implante_clase": 2
}

img_bgr = cv2.imread('path_to_image.jpg')
results = run_pipeline(img_bgr, params)

cv2.imshow('Bone Edges', results['edges'])
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Parameter Configuration

* `log_gain`: Gain factor for logarithmic preprocessing.
* `prep_mode`: Choose 'log' or 'none' for preprocessing.
* `median_ksize`: Kernel size for median filtering.
* `block_size`: Neighborhood size for adaptive thresholding.
* `C`: Compensation constant for adaptive thresholding.
* `n_classes`: Number of classes for Multi-Otsu segmentation.
* `min_area`: Minimum area threshold for morphological filtering.
* `implante_clase`: Class label corresponding to the implant.

## Outputs

The pipeline returns a dictionary with intermediate and final outputs:

* `gray`: Preprocessed grayscale image.
* `adaptive`: Result of adaptive thresholding.
* `labeled`: Multi-class segmentation.
* `cleaned`: Segmentation after morphological filtering and hole filling.
* `edges`: Final bone edge map.
* `thresholds`: Threshold values computed by Multi-Otsu segmentation.

## Contributions and Issues

Feel free to open an issue for bug reports, feature requests, or submit a pull request if you'd like to contribute improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
