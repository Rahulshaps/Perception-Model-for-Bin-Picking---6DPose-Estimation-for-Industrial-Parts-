"""
Author: Rahul Sha Pathepur Shankar, Vaishnav Raja & Reza Farrokhi Saray
CS 5330 Pattern Recognition and Computer Vision - Spring 2025 - Final Project

This code is part of the 6D pose estimation for the Industrial Plentopic Dataset (IPD)
This code is prepares the dataset for training YOLOv11 for object detection by converting
the dataset from BOP format to YOLO format 
"""

import os
import sys
import json
import shutil
from PIL import Image

# object ID to filter in the dataset as we training each object separately
objId = 0 

# setting up the path to the script directory for importing 
trainPbrPath = "./Dataset/train_pbr"
outputPath = "./Dataset/yolo/output"

# creating the path to the script directory if it does not exist
if not os.path.exists(outputPath):
    print("Creating output folder:", outputPath) 
    os.makedirs(outputPath)

imagesFolder = outputPath + "/images"
labelsFolder = outputPath + "/labels"

if not os.path.exists(imagesFolder):
    print("Creating images folder")
    os.makedirs(imagesFolder)
if not os.path.exists(labelsFolder):
    print("Creating labels folder")
    os.makedirs(labelsFolder)

# setting up the camera names to filter the dataset as we have multiple cameras (number = 3) 
cameras = ["rgb_cam1", "rgb_cam2", "rgb_cam3"]

# Going throught the dataset and copying the images and labels to the output folder
scenes = os.listdir(trainPbrPath)
# sorting the scenes to have a order
scenes.sort() 

# looping through all scenes
for scene in scenes:
    sceneDir = trainPbrPath + "/" + scene
    if not os.path.isdir(sceneDir):
        continue

    print("\nProcessing scene:", scene)

    for cam in cameras:
        # building the paths to the RGB images and JSON files
        rgbDir = sceneDir + "/" + cam
        gtJson     = sceneDir + "/scene_gt_"     + cam.split("_")[-1] + ".json"
        infoJson   = sceneDir + "/scene_gt_info_" + cam.split("_")[-1] + ".json"

        # checking if the directories and files exist
        if not os.path.isdir(rgbDir):
            print("  Missing folder:", rgbDir)
            continue
        if not os.path.isfile(gtJson) or not os.path.isfile(infoJson):
            print("  Missing JSON for", cam)
            continue

        # loading JSON data
        gtData   = json.load(open(gtJson))
        infoData = json.load(open(infoJson))
        imgCount = len(gtData)

        # checking if the number of images is the same as the number of ground truth boxes
        for i in range(imgCount):
            key = str(i)
            # finding image file (jpg or png)
            jpg = f"{rgbDir}/{i:06d}.jpg"
            png = f"{rgbDir}/{i:06d}.png"
            if os.path.exists(jpg):
                imgFile = jpg
            elif os.path.exists(png):
                imgFile = png
            else:
                continue 

            # getting ground truth lists
            if key not in gtData or key not in infoData:
                continue

            # looping through all objects in the image
            # and getting the bounding boxes for the object ID
            bboxes = []
            for objInfo, objGt in zip(infoData[key], gtData[key]):
                if objGt["obj_id"] == objId and objInfo["visib_fract"] > 0:
                    bboxes.append(objInfo["bbox_obj"])

            if len(bboxes) == 0:
                continue  

            # copying image to new folder with a new name
            newImgName = f"{scene}_{cam}_{i:06d}.jpg"
            newImgPath = imagesFolder + "/" + newImgName
            shutil.copy(imgFile, newImgPath)
            print("  Copied image:", newImgName)

            # opening image to get size
            imgW, imgH = Image.open(imgFile).size

            # writing YOLO label file
            labelName = newImgName.replace(".jpg", ".txt")
            labelPath = labelsFolder + "/" + labelName
            with open(labelPath, "w") as f:
                for box in bboxes:
                    x, y, w, h = box
                    # converting to YOLO format
                    x_center = (x + w/2) / imgW
                    y_center = (y + h/2) / imgH
                    w_norm   = w / imgW
                    h_norm   = h / imgH
                    # writing to label file
                    # YOLO format: <object-class> <x_center> <y_center> <width> <height>
                    line = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
                    f.write(line)

            print("Wrote label:", labelName)

print("\nAll done! Check your", imagesFolder, "and", labelsFolder)