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
  
 each subfolder of ```sharks``` contains images of the shark specie specified in the folder name (thus the folder name is acting as a label for each class).
