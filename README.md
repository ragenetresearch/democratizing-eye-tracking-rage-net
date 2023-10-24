# Democratizing Eye-Tracking? RAGE-net: Appearance-Based Gaze Estimation with Improved Attention Branch

## About

This is the official Github repository of the research paper [Democratizing Eye-Tracking? RAGE-net: Appearance-Based Gaze Estimation with Improved Attention Branch]().
The contents of this paper subsume two primary topics. First is the proposal of RAGE-net (Residual Apperance-based Gaze Estimation Network), a novel convolutional neural network for calibrationless prediction of the user's gaze point on the screen, by using data from a webcam. With the angular error of 4.08° in the MPIIFaceGaze dataset, RAGE-net outperforms state-of-the-art calibrationless appearance-based models and uses a considerably smaller number of parameters. The second subject of the paper is an applicability analysis of the apperance-based model - investigation of how the error of the model (higher compared to established feature-based infrared eye trackers) translates into gaze visualizations, and how factors of the environment such as illumination, position of the camera, distance from the screen or wearing glasses practically impact the resulting gaze estimation.

#### Table of Contents
* [Paper Citation](#a-citation)
* [Installation](#a-installation)
* [Data normalization](#a-data-normalization)
* [Gaze estimation](#a-gaze-estimation)
* [Hypothesis testing](#a-hypothesis-testing)
* [Visualization](#a-visualization)
* [Authors](#a-authors)
* [License](#a-license)

## <a name="a-citation"> Paper Citation
[Place for a bibtex-format paper citation]

## <a name="a-installation"> Installation
Due to their size, the trained model and the dataset of participants collected to evaluate the impact of factors of the environment (Experiment 3 in the paper) are [shared via Google Drive](https://drive.google.com/drive/folders/1RHs7xGCD-k13N2YD2P0-54d0tmD7_XKy?usp=share_link). To assure compatibility and replicability, models for face detection and estimation of 2D facial landmarks from the utilized OpenCV and dlib libraries respectively, are also backed up on drive. After ckecking out this repository, to prepare the environment, neccessary drivers and to install all requirements in the expected directories, follow the [prerequisites guide](./Docs/Prerequisites.md).

The drive directory contains:
* `rn_w_attention__tf_model` - RAGE-net model trained on the MPIIFaceGaze dataset
* `Study_2.zip` - complete dataset from Experiment 3 (webcam images of 30 participants gazing at fixed points under 6 varied environmental conditions)
* `OpenCvDNN` - OpenCV Caffe model used for face detection
* `ShapePredictors` - [dlib](http://dlib.net/) shape predictor model used for estimation of 2D facial landmarks

Repository structure:
* Dataset - data obtain from the experiment (see instructions above for how to import from Google Drive)
* Docs - supporting documents (see instructions above for how to import from Google Drive)
* Models - prediction models and their source code
* Notebooks - Jupyter source code for prediction, hypothesis testing and generation of visualizations
* Shape predictors - dlib model for face detection (see instructions above for how to import from Google Drive)
* Utils - helper script files

## <a name="a-data-normalization"> Data normalization
The proposed network entails an input data normalization pipeline, where raw input images from the webcam are preprocessed to be transformed into a form expected by the network. See Jupyter notebook [Preprocess](./Notebooks/Study2/Preprocess.ipynb) to replicate the normalization.

## <a name="a-gaze-estimation"> Gaze estimation
TBA
[Predict](./Notebooks/Study2/Predict.ipynb)

## <a name="a-hypothesis-testing"> Hypothesis testing
TBA
[Test hypothesis](./Notebooks/Study2/Test-hypothesis.ipynb)

## <a name="a-visualization"> Visualization
TBA
[Visualize](./Notebooks/Study2/Visualize.ipynb)

## <a name="a-authors"> Authors
TBA

## <a name="a-license"> License

This work is licensed under a
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">
Creative Commons Attribution-NonCommercial 4.0 International License.
</a>

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/" style="margin-left: 8rem">
<img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" />
</a>




