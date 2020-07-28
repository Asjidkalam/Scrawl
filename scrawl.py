#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import cv2
import pickle
import pytesseract
import numpy as np 
from keras.models import load_model

# load the model
model = load_model('models/handjob.h5')     # HTR Model trained with EMNIST dataset.

# Import your tesseract executable here (eg: '/usr/bin/tesseract' for linux)
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

background = Image.open("out/bg.png")
bgSize = background.width
gap, _ = 0, 0 
allowedChars = 'qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM,.-?!() 1234567890'

def detection(handwriting_data):
    img = cv2.imread(handwriting_data) 

    # image -> gray scale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # OTSU threshold 
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 
    # Dilation on the threshold image 
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 
    # Contours 
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
                                                    cv2.CHAIN_APPROX_NONE) 
    im2 = img.copy() 

    i = 0 
    os.mkdir('out')
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        #rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped = im2[y:y + h, x:x + w]
        cv2.imwrite("out/"+str(i)+".png", cropped)
        i+=1

    print("[+] Identification completed.")

def writeData():
    global gap, _
    if char == '\n':
        pass
    else:
        char.lower()
        cases = Image.open("out/%s.png" % char)
        background.paste(cases, (gap, _))
        size = cases.width
        gap += size
        del cases

def letterWrite(word):
    global gap, _
    if gap > bgSize - 95 * (len(word)):
        gap = 0
        _ += 200
    for letter in word:
        if letter in allowedChars:
            if letter.islower():
                pass
            elif letter.isupper():
                letter = letter.lower()
                letter += 'upper'
            elif letter == '.':
                letter = "fullstop"
            elif letter == '!':
                letter = 'exclamation'
            elif letter == '?':
                letter = 'question'
            elif letter == ',':
                letter = 'comma'
            elif letter == '(':
                letter = 'braketop'
            elif letter == ')':
                letter = 'braketcl'
            elif letter == '-':
                letter = 'hiphen'
            writeData(letter)
        else:
            print("[X] Character " + letter +" is not allowed.")
            exit()

# Splits the input to individual words and write
def words(Input):
    wordlist = Input.split(' ')
    for i in wordlist:
        letterWrite(i)
        writeData('space')

def recognize():
    # Iterating through each images, predicting and renaming
    for imageName in os.listdir('out/'):
        img_dir = 'out/'
        print("Image name: " + img_dir + imageName)
        try:
            with open(os.path.abspath(os.path.join(os.getcwd(), "data/mapping.pkl"))) as f:
                mapping = pickle.load(f)
        except Exception as e:
            print(e)
            mapping = ['0', '1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

        img = cv2.imread(img_dir + imageName, 0)
        # Might not be the most efficent way, but it will do for now.

        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th3 = cv2.subtract(255, th3)
        pred_img = th3
        pred_img = cv2.resize(pred_img, (28, 28))
        pred_img = pred_img / 255.0
        pred_img = pred_img.reshape(1,784)

        # Predict
        prediction = mapping[model.predict_classes(pred_img)[0]]
        print("\n\nPredicted Value : {}".format(prediction))

        # Rename image
        if prediction.isupper():
            prediction = prediction.lower()
            prediction += 'upper'
            os.rename(img_dir + imageName, img_dir + prediction + '.jpg')
        else:
            os.rename(img_dir + imageName, img_dir + prediction + '.jpg')

        print("[+] Recognition completed. Although the classifier may not have predicted the exact characters. Check the `out` directory and rename imgs if needed.")
        wait = input("[!] Continue? (Y/n)")
        if wait == "Y" or wait == "y" or wait == "":
            writeText()
        else:
            print("Bye!")

def writeText(input_txt):
    try:
        with open(input_txt, 'r') as file:
            data = file.read().rstrip('\n')

        ln = len(data)
        nn = ln // 600
        chunks, chunk_size = ln, ln // (nn + 1)
        p = [data[i:i + chunk_size] for i in range(0, chunks, chunk_size)]

        for i in range(0, len(p)):
            words(p[i])
            writeData('\n')
            background.save('%doutt.png' % i)
            background1 = Image.open("out/bg.png")
            background = background1
            gap = 0
            _ = 0
    except ValueError as err:
        print("[X] {} - Try Again".format(err))

if __name__ == '__main__':
    print("Scrawl Handwritten Project")

    # TODO: parse arguments like input image, input text etc etc
    # ! no convert to pdf as of now.!
    argParser = argparse.ArgumentParser(description="Convert text to handwritten images.")
    argParser.add_argument('-hw', '--handwriting', action='store', dest='hw_data', required=True, help='Provide the alphabetical handwriting image with extension')
    argParser.add_argument('-t', '--text', action='store', dest='input_txt', required=True, help='Provide the text to convert')
    argParser.add_argument('--version', action='version', version='%(prog)s 0.1')
    userParams = argParser.parse_args()

    # Perform handwriting countour detection and split each letters into separate images
    detection(userParams.hw_data)

    # Recognize each letters/images and rename the images accordingly
    recognize()

    # Write the text using the recognized characters
    writeText(userParams.input_txt)
    