<img src="../assets/shark_annotation.png">

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
 
 The installation procedure can be found on the [project's Github page](https://github.com/gcgrossi/google-vision-wrapper), together with the links to properly setup the Google Vision API.
 
 Once everything is setup, we can move on to the juicy part: Python.
 
 ### Python Script
 in the root directory I create a file named  ```annotate_images.py``` and start by importing
 
 #### Imports
 
  ```python
 
import os
import cv2
import random

import numpy as np
import pandas as pd
 
 from csv import writer
 from gvision import GVisionAPI
 
 ```
 
the python package of ```google-vision-wrapper``` is ```gvision``` and from it we import the ```GVisionAPI``` class. We will need the other packages in the future. From here we proceed by writing some helper functions.
 
 #### Helper Functions
 
 ```python
 
file_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def list_files(indir=os.getcwd(),valid_extensions=file_extensions):
    for (rootdir,dirs,files) in os.walk(indir):
        for filename in files:
            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()
            
            # check to see if the file is an image and should be processed
            if valid_extensions is None or ext.endswith(valid_extensions):
                
                # construct the path to the image and yield it
                imagePath = os.path.join(rootdir, filename)
                
                # yield the path
                yield imagePath
            
    return
 ```
 
This function crawls the input directory and checks if the extension of the image is valid. Using the python method ```yield``` it returns an iterator, over which we will loop later on in the annotation face.

```python 

def load_annotation(annotation_file):
    
    # init DataFrame columns
    header_list = ["image","label","tlx","tly","brx","bry"]

    # check if the annotation file exists
    if os.path.isfile(annotation_file) : 
        # if yes -> read as DataFrame
        df = pd.read_csv(annotation_file,names=header_list)
    else:
        # if no -> create an empty DataFrame
        df = pd.DataFrame(columns = header_list)
    
    return df
 
 ``` 
 
This function opens an input csv file and returns a pandas DataFrame with the stored info.

```python
 
def add_annotation(annotation_csv,annotation_list):

    with open(annotation_csv, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(annotation_list)
    return
 
 ```
 
 This function opens an input csv file and appends an input list (which should be correctly comma separated) to the existing file.
 
 
 
