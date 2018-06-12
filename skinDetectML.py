#!/usr/bin/python
import os,glob,argparse,sys
import cv2
import numpy as np
import time

imageDir = 'Face_Dataset/Pratheepan_Dataset/'
groundTruthDir = 'Face_Dataset/Ground_Truth/'
  
def getFileNames():
  types = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
  imageFiles = []
  gtFiles = []
  for filetype in types:

    # there are two folders inside the parent directory
    imageFiles += glob.glob(imageDir + '*/*' + filetype)
    gtFiles += glob.glob(groundTruthDir + '*/*' + filetype)
      
  imageFiles.sort()
  gtFiles.sort()

  return imageFiles, gtFiles


def prepareTrainingData(imageFiles, gtFiles, colorspace):
    
    # Initialize the array for storing the training samples
    X = np.zeros((1,3))
    Y = np.zeros((1,1))
    for i in range(len(imageFiles)):

      # Read the image and it's mask 
      image = cv2.imread(imageFiles[i])
      mask = cv2.imread(gtFiles[i])

      w,h,ch = image.shape

      # Convert the images to 2-D arrays
      imageArray = image.reshape(w*h, ch)
      maskArray = mask.reshape(w*h, ch)
      
      # Find the pixels in image which are skin and non skin using the mask
      skinPixels = imageArray[maskArray.any(axis = 1)]
      nonSkinPixels = imageArray[~maskArray.any(axis = 1)]
      
      # create an array for the labels 
      #  0 - nonSkin, 1 - skin 
      skinLabels = np.ones((skinPixels.shape[0],1))
      nonSkinLabels = np.zeros((nonSkinPixels.shape[0],1))
      
      # Concatenate present data with previous data
      X = np.r_[X, skinPixels, nonSkinPixels]
      Y = np.r_[Y, skinLabels, nonSkinLabels]
      
    # Convert to required color space
    if colorspace == 'y':  
      X = cv2.cvtColor(np.array([X], dtype=np.uint8), cv2.COLOR_BGR2YCrCb)[0]

    # Convert all data to float and range (0, 1)
    X = np.float32(X)/255.0
    Y = np.int32(Y)
    
    return X, Y


if __name__ == '__main__':
  
  print("\n\nNOTE : Please download the data from the link below if not already done.\n \
        http://cs-chan.com/downloads_skin_dataset.html\n \
    ")

  if not os.path.exists(imageDir):
    print("\nPlease provide the correct path to data")
    sys.exit()

  ap = argparse.ArgumentParser()
  ap.add_argument("-f", "--filename", help="Path to the image")
  ap.add_argument("-c", "--colorspace",  help="color space to use")
  args = vars(ap.parse_args())

  # Default file name & colorspace
  testFileName = "images/hillary_clinton.jpg"
  colorspace = 'y'

  if(args["filename"]):
    testFileName = args["filename"]
  if(args["colorspace"]):
    colorspace = args["colorspace"]

  imageFiles, gtFiles = getFileNames()

  print("Training . . .")
  ####################   TRAINING   ########################
  Start = time.time()
  X, Y = prepareTrainingData(imageFiles, gtFiles, colorspace)

  # train naive bayes
  nbc = cv2.ml.NormalBayesClassifier_create()
  nbc.train(X, cv2.ml.ROW_SAMPLE, Y)
  End = time.time()
  trainTime = End - Start
  print("Training time = "+str(trainTime)+" Seconds\n")

  ####################   TESTING   ########################
  Start = time.time()
  print("Testing . . .")
  testImageOriginal = cv2.imread(testFileName)

  # Convert to required color space
  if colorspace == 'y':
    print( "USING YCrCb COLOR SPACE")
    testImage = cv2.cvtColor(testImageOriginal, cv2.COLOR_BGR2YCrCb)
  if colorspace == 'b':
    print( "USING BGR COLOR SPACE")
    testImage = testImageOriginal
    
  testImage = cv2.GaussianBlur(testImage, (7, 7), 0)
  testImage = np.float32(testImage)/255.0

  w,h,ch = testImage.shape
  testImageArray = testImage.reshape(w*h, -1)

  testX = np.float32(testImageArray)
  _, predLabel= nbc.predict(testX)

  outputMask = predLabel.reshape(w,h)
  imgDst = np.copy(testImageOriginal)

  imgDst[outputMask == 0] = 0
  End = time.time()
  testTime = End - Start
  print("Testing time = "+str(testTime)+" Seconds\n")

  ################# SAVE  #################
  testFileName = os.path.basename(testFileName)
  if(colorspace == 'y'):
    write_name = 'results/NaiveBayesClassifier/'+testFileName+'_skin_NBC_YCbCr.jpg'
    cv2.imwrite(write_name, imgDst);
  else:
    write_name = 'results/NaiveBayesClassifier/'+testFileName+'_skin_NBC_BGR.jpg'
    cv2.imwrite(write_name, imgDst);

  