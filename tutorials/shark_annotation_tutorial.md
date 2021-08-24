<image src="../assets/shark_annotation.png">

# Shark images annotation using Python 🐍, Google Vision API and google-vision-wrapper 👁

### Problem statement 
I am currently setting up a bounding box regression model to construct a shark detector, which will be able to correctly spot and classify different shark species in an image. I have about 2'000 shark images organized in this folder structure:
 
 ```
 ./
  ├─── sharks
       ├───great_white_shark
       ├───hammerhead_shark
       ├───mako
       ├───not_shark
       ├───tiger_shark
       └───whale_shark
 ```
  
 each subfolder of ```sharks``` contains images of the shark specie specified in the folder name (thus the folder name is acting as a label for each class). I need to write down (annotate) some information for each image:
 
 1. file name
 2. label
 3. (x,y) coordinate of bounding box's top-left corner
 4. (x,y) coordinate of bounding box's bottom right corner
 
 and dump it in a csv file. Usually this is done by hand, using some annotating tool (online or offline), but today we are going to exploit the Google Vision API, along with a python package ```google-vision-wrapper```

### Setup Google Vision API and google-vision-wrapper
Google Vision API is a service provided by Google, which enables to send a request to their API and perform different kind of computer vision tasks: face detection, label detection, object detection as well as optical character recognition (OCR). Today we are going to use the object detection feature.  
 
```google-vision-wrapper``` is an handy python package that simplifies the interface and the request to the API in an effective way. The working environment requires to have a Google Cloud Platform Account, the Google Vision API enabled and the ```google-vision-wrapper``` package installed via:
 
 ```pip install google-vision-wrapper```
 
 
