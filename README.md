# Democratizing Eye-Tracking? RAGE-net: Appearance-Based Gaze Estimation with Improved Attention Branch

## About

This is the official Github repository of the research paper [Democratizing Eye-Tracking? RAGE-net: Appearance-Based Gaze Estimation with Improved Attention Branch].
The contents of this paper subsume two primary topics. First is the proposal of RAGE-net (Residual Apperance-based Gaze Estimation Network), a novel convolutional neural network for calibrationless prediction of the user's gaze point on the screen, by using data from a webcam. With the angular error of 4.08° in the MPIIFaceGaze dataset, RAGE-net outperforms state-of-the-art calibrationless appearance-based models and uses a considerably smaller number of parameters. The second subject of the paper is an applicability analysis of the apperance-based model - investigation of how the error of the model (higher compared to established feature-based infrared eye trackers) translates into gaze visualizations, and how factors of the environment such as illumination, position of the camera, distance from the screen or wearing glasses practically impact the resulting gaze estimation.

## Study 2

- [Prerequisities](./Docs/Prerequisites.md)

**Notebooks:**

1. [Preprocess](./Notebooks/Study2/Preprocess.ipynb)
2. [Predict](./Notebooks/Study2/Predict.ipynb)
3. [Test hypothesis](./Notebooks/Study2/Test-hypothesis.ipynb)
4. [Visualize](./Notebooks/Study2/Visualize.ipynb)
