# Democratizing Eye-Tracking? RAGE-net: Appearance-Based Gaze Estimation with Improved Attention Branch

## About

This is the official Github repository of the research paper [Democratizing Eye-Tracking? RAGE-net: Appearance-Based Gaze Estimation with Improved Attention Branch]().
The contents of this paper subsume two primary topics. First is the proposal of RAGE-net (Residual Apperance-based Gaze Estimation Network), a novel convolutional neural network for calibrationless prediction of the user's gaze point on the screen, based on image data obtained from a webcam. With the angular error of 4.08° in the [MPIIFaceGaze](https://www.perceptualui.org/research/datasets/MPIIFaceGaze/) dataset, RAGE-net outperforms state-of-the-art calibrationless appearance-based models and uses a considerably smaller number of parameters.

The second subject of the paper extends to an applicability analysis of the apperance-based model - investigation of how the angular error of the model (which is higher compared to established feature-based gaze estimation utilized by dedicated infrared eye trackers) translates into reliability of its gaze visualizations, and how the factors of the environment such as illumination, position of the camera, distance from the screen or whether users are wearing glasses practically impact the resulting gaze estimation.

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
Due to their size, the trained model and the dataset of participants collected to evaluate the impact of factors of the environment (Experiment 3 in the paper) are [shared via Google Drive](https://drive.google.com/drive/folders/1RHs7xGCD-k13N2YD2P0-54d0tmD7_XKy?usp=share_link). To assure compatibility and replicability, models for face detection and estimation of 2D facial landmarks from the utilized OpenCV and dlib libraries respectively, are also backed up on the drive. After cloning the contents of this repository, to prepare the environment, the neccessary drivers and to install all requirements in the expected directories, follow the [prerequisites guide](./Docs/Prerequisites.md).

The drive directory contains:
* `rn_w_attention__tf_model` - RAGE-net model trained on the [MPIIFaceGaze](https://www.perceptualui.org/research/datasets/MPIIFaceGaze/) dataset
* `Study_2.zip` - complete dataset from Experiment 3 (webcam images of 30 participants gazing at fixed points under 6 varied environmental conditions)
* `OpenCvDNN` - OpenCV Caffe model used for face detection
* `ShapePredictors` - [dlib](http://dlib.net/) shape predictor model used for estimation of 2D facial landmarks

Repository structure:
* Dataset - data obtained from the experiment (see instructions above for how to import from Google Drive)
* Docs - supporting documents
* Models - prediction models and their source code (see instructions above for how to import from Google Drive)
* Notebooks - Jupyter source code for prediction, hypothesis testing and generation of visualizations
* Shape predictors - dlib model for face detection (see instructions above for how to import from Google Drive)
* Utils - helper script files

## <a name="a-data-normalization"> Data normalization
The proposed network design entails an input data normalization pipeline, where raw input images from the webcam are preprocessed to be transformed into a form expected by the network. See Jupyter notebook [Preprocess](./Notebooks/Study2/Preprocess.ipynb) to replicate the normalization.

## <a name="a-gaze-estimation"> Gaze estimation
Gaze estimation with the RAGE-net model can be replicated with the [Predict](./Notebooks/Study2/Predict.ipynb) notebook.

## <a name="a-hypothesis-testing"> Hypothesis testing
Stated hypotheses about the effect of environmental factors on RAGE-net model gaze estimations are statistically tested, and diagrams illustrating these statistical tests are plotted in the [Test hypothesis](./Notebooks/Study2/Test-hypothesis.ipynb) notebook.

## <a name="a-visualization"> Visualization
Visualizations of gaze tracking accuracy (positions of gaze points on the screen) are generated in the [Visualize](./Notebooks/Study2/Visualize.ipynb) notebook.

## <a name="a-authors"> Authors

### General contact

Email: ragenet.research([AT])gmail.com

### Eduard Kuric
He received his PhD degree in computer science from the [Faculty of Informatics and Information Technologies](https://www.fiit.stuba.sk/), [Slovak University of Technology in Bratislava](https://www.stuba.sk/). He is a researcher and assistant professor at the same university. His research interests include human-computer interaction, user modeling, personalized web-based systems, and machine learning. Eduard is also the head of the UX Research Department at [UXtweak](https://www.uxtweak.com) and the founder of [UXtweak Research](https://www.uxtweak.com).
- Email: eduard.kuric([AT])stuba.sk
- [LinkedIn](https://www.linkedin.com/in/eduard-kuric-b7141280/)
- [Google Scholar](https://scholar.google.com/citations?user=MwjpNoAAAAAJ&hl=en&oi=ao)

### Peter Demcak
He received his master’s degree in computer science from the [Faculty of Informatics and Information Technologies](https://www.fiit.stuba.sk/), [Slovak University of Technology in Bratislava](https://www.stuba.sk/). He is a researcher with background in software engineering, whose current topics of interest involve user behavior, human-computer interaction, UX research methods & design practices, and machine learning. Currently occupies the position of a scientific and user experience researcher at [UXtweak Research](https://www.uxtweak.com/), with focus on research that supports work of UX professionals.
- [LinkedIn](https://www.linkedin.com/in/giang-nguyen-3307b8b/)
- [Google Scholar](https://scholar.google.com/citations?hl=en&user=IEmgzZkAAAAJ)
- Email: peter.demcak([AT])uxtweak.com

### Giang Nguyen
She is a senior researcher and associate professor at the [Faculty of Informatics and Information Technologies](https://www.fiit.stuba.sk/), [Slovak University of Technology in Bratislava](https://www.stuba.sk/). She focuses on machine learning, deep learning, soft computing, and security and reliability. She is also a reviewer and editor for Web of Science journals and member of program committee, editor, organizator for international conferences.
- [LinkedIn](https://www.linkedin.com/in/giang-nguyen-3307b8b/)
- [Google Scholar](https://scholar.google.com/citations?hl=en&user=IEmgzZkAAAAJ)
- Email: giang.nguyen([AT])stuba.sk

### Jozef Majzel
He holds a master’s degree in computer science from the [Faculty of Informatics and Information Technologies](https://www.fiit.stuba.sk/), [Slovak University of Technology in Bratislava](https://www.stuba.sk/). He specializes in areas such as user behavior, UX research methods and design principles, as well as computer vision and deep learning. Presently holds the role of a scientific and user experience researcher at [UXtweak Research](https://www.uxtweak.com/), where he focuses on user behavior analytics.
- [LinkedIn](www.linkedin.com/in/jozef-majzel)
- [Google Scholar](https://scholar.google.com/citations?user=ywuWTh0AAAAJ&hl)
- Email: xmajzel@stuba.sk

## <a name="a-license"> License

This work is licensed under a
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">
Creative Commons Attribution-NonCommercial 4.0 International License.
</a>

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/" style="margin-left: 8rem">
<img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" />
</a>




