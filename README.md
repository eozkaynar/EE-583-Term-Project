EE583 Term Project:<br/>ViViT for 3D classification
------------------------------------------------------------------------------
The Vision Video Transformer (ViViT) is a state-of-the-art architecture designed for video classification tasks, leveraging the power of Transformer models to process spatio-temporal data effectively. Unlike traditional convolutional neural networks (CNNs), ViViT employs attention mechanisms to capture complex relationships across spatial and temporal dimensions, offering enhanced representational capacity. By tokenizing video data into spatio-temporal patches and processing them through transformer layers, ViViT provides a robust framework for analyzing dynamic visual information, making it particularly suitable for applications in medical imaging and action recognition [1].
In this work ViViT model-1 is implemented.

Dataset
-------
OrganMNIST3D is part of the MedMNIST [1] collection, specifically designed for 3D medical image classification. This dataset consists of 3D CT (Computed Tomography) images, each represented as a 3D array with dimensions of 28×28×28. The organ classes represented in the dataset include the liver, right kidney, left kidney, right femur, left femur, bladder, heart, right lung, left lung, spleen, and pancreas. The dataset is divided into three subsets: a training set with 971 samples, a validation set with 161 samples, and a test set with 610 samples[2].



| Normal                                 | Low Ejection Fraction                  | Arrhythmia                             |
| ------                                 | ---------------------                  | ----------                             |
| ![](docs/media/0X10A28877E97DF540.gif) | ![](docs/media/0X129133A90A61A59D.gif) | ![](docs/media/0X132C1E8DBB715D1D.gif) |
| ![](docs/media/0X1167650B8BEFF863.gif) | ![](docs/media/0X13CE2039E2D706A.gif ) | ![](docs/media/0X18BA5512BE5D6FFA.gif) |
| ![](docs/media/0X148FFCBF4D0C398F.gif) | ![](docs/media/0X16FC9AA0AD5D8136.gif) | ![](docs/media/0X1E12EEE43FD913E5.gif) |

The code implementation in this study builds upon the robust framework established by Gosthipaty and Thakur [3], whose work provided a foundational basis for the design and experimentation of the ViViT model.

Installation
------------

First, clone this repository and enter the directory by running:

    git clone https://github.com/echonet/dynamic.git
    cd dynamic

The model is implemented for Python 3, and depends on the following packages:
  - NumPy
  - PyTorch
  - Torchvision
  - OpenCV
  - skimage
  - sklearn
  - tqdm

Code dependencies can be installed using

    pip install -r 'requirements.txt'

Usage
-----

### Running Code

EchoNet-Dynamic has three main components: segmenting the left ventricle, predicting ejection fraction from subsampled clips, and assessing 

#### Classification with ViViT

    echonet segmentation --save_video

This creates a directory named `output/vivit/`, which will contain
  - log.csv: training and validation losses
  - best.pt: checkpoint of weights for the model with the lowest validation loss
  - size.csv: estimated size of left ventricle for each frame and indicator for beginning of beat
  - videos: directory containing videos with segmentation overlay

#### Classification with 3D CNN model

  echonet video

This creates a directory named `output/cnn3d/`, which will contain
  - log.csv: training and validation losses
  - best.pt: checkpoint of weights for the model with the lowest validation loss
  - test_predictions.csv: ejection fraction prediction for subsampled clips

### Hyperparameter Test

Hyperparameter test can be run via `scripts/hyperparameter_test.sh`. The results of the hyperparameter tuning experiment will be stored in the directory named hyperparameter_outputs
The hyperparameter tuning experiment was conducted to evaluate the performance of the model across various configurations. The tested hyperparameters include three learning rates (10−410−4, 10−310−3, and 10−510−5), three patch sizes ("4,4,4", "6,6,6", and "8,8,8"), and three projection dimensions (64, 128, and 256). Additionally, two different numbers of attention heads (4 and 8) and two different numbers of transformer layers (6 and 8) were explored. These combinations aim to identify the optimal configuration that maximizes model accuracy while maintaining computational efficiency.



Citation
------------
[1] Arnab, A., Dehghani, M., Heigold, G., Sun, C., Lučić, M., \& Schmid, C. (2021). ViViT: A Video Vision Transformer. arXiv. https://arxiv.org/abs/2103.15691

[2] Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni. Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification." Scientific Data, 2023.

[3] Keras Team, "Video Vision Transformer (ViViT)," Keras.io, Accessed: January 18, 2025. [Online]. Available: https://keras.io/examples/vision/vivit/

