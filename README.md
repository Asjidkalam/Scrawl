# Scrawl Project

Automating assingments, homeworks and projects by text to user-handwriting conversion using machine learning HTR system 😄.

This project uses the **keras** deeplearning library to recognize the user images, **Tesseract-OCR** and **OpenCV-Python** to detect and to convert images as notes. Here, the handwritten character recognition is implemented using the **MNIST** dataset.

The MNIST dataset is a set of handwritten character digits derived from the [NIST Special Database 19](https://www.nist.gov/srd/nist-special-database-19)  and converted to a 28x28 pixel image format and dataset structure.

![Sample Image from dataset](https://miro.medium.com/max/1200/1*xsUfUB4g4171IUqSzcwY6Q.png)


## Introduction

For those starting in the Optical Character Recognition (OCR) environment, here's some brief context:
Basically, the algorithm takes an image (image of a handwritten character) as an input and outputs the likelihood that the image belongs to different classes (the machine-encoded character, a-zA-Z). So, in our case, the goal is to take an image of a handwritten character and determine what that alphabet is, using a trained model (*HTR*).

For many years, HTR systems have used the Hidden Markov Models (HMM) for the transcription task, but recently, through Deep Learning, the Convolutional Recurrent Neural Networks (CRNN) approach has been used to overcome some limitations of HMM.


<div style="text-align:center"><img align="center" src="https://i.imgur.com/g2XKwxW.png" alt="CRNN"/></div>

*Overview of CRNN*


## Training process 

The workflow is divided into 3 steps:
* The input image is fed into the CNN layers to extract features. The output is feature map.
* Through the implementation of Long Short-Term Memory (LSTM), the RNN is able to propagate information over longer distances and provide more robust features to training.
* With RNN Connectionist Temporal Classification (CTC), calculate the loss value and also decodes into the final text.

Finally, all training and predictions were conducted on the Google Colaboratory (Colab) platform. By default, the platform offers Linux operating system, with 12GB ram and Nvidia Tesla T4 GPU 16GB memory (thank you so much, Google ❤).

Tesseract-OCR is used to detect the contours and convert each letter to individual images along with OpenCV. OpenCV then uses the detected and recognized image data to paste each letters into a background image that we supplied.

This project uses code created/inspired by many other repositories and projects. Links to all of them will be given in the *references*.


## Pre-requisites:
- python 3.5
- [Install Tesseract: an open source text recognition (OCR) Engine](https://github.com/tesseract-ocr/tessdoc/blob/master/Home.md)
- all dependencies from `requirements.txt`


## Installation:

- Clone the repository
```sh
git clone https://github.com/Asjidkalam/scrawl/
```

- Install the necessary dependencies
```sh
pip install -r requirements.txt
```

- Change the path to `tesseract-ocr`'s executable in [this](https://github.com/Asjidkalam/Scrawl/blob/master/scrawl.py#L17) line.


## Usage:

* Displays the current version
```
python3 scrawl.py --version
```

* Use the handwriting data image(`-hw/--handwriting`) and text data(`-t/--text`) to convert
```
python3 scrawl.py -hw my_handwriting.jpg -t test_data.txt
```

* Use the images used earlier from `font/` directory
```
python3 scrawl.py --usetrained -t test_data.txt
```


## References:

* http://yann.lecun.com/exdb/publis/pdf/matan-90.pdf 
* [Sharanya Mukherjee](https://www.linkedin.com/in/sharanya-mukherjee-73a2061a0/)'s Text to handwriting conversion [project](https://github.com/sharanya02/Text-file-to-handwritten-pdf-file). 
*  https://medium.com/the-andela-way/applying-machine-learning-to-recognize-handwritten-characters-babcd4b8d705
*  https://medium.com/@arthurflor23/handwritten-text-recognition-using-tensorflow-2-0-f4352b7afe16
*  https://github.com/srijan14/keras-handwritten-character-recognition
*  https://medium.com/programmersclub/training-a-deep-learning-model-on-handwritten-characters-using-keras-4bad2124a6e1
*  P. Voigtlaender, P. Doetsch, and H. Ney, “Handwriting recognition with large multidimensional long short-term memory recurrent neural networks”, in 15th International Conference on Frontiers in Handwriting Recognition (ICFHR)
*  U.-V. Marti and H. Bunke, “The iam-database: An english sentence database for offline handwriting recognition”, International Journal on Document Analysis and Recognition
*  M. Sonkusare and N. Sahu, “A survey on handwritten character recognition (hcr) techniques for english alphabets”, Advances in Vision Computing: An International Journal, vol. 3


🍰

