# Traffic Sign Recognition

Traffic sign recognition is a part of an advanced driver assistance system which can recognize road signs
and pass on the corresponding information for further vehicle control.
The problem is broken into 2 parts Sign Detection and Sign Classification. In detection, we will detect the
coordinates and size of a signboard in the image and in classification, we try to categorize what the signal
represents or what class this signal belongs.

Find the report for the project here:
[Report](https://github.com/amrish1222/traffic_sign_detection/blob/master/Report.pdf)

## Sample Output

![tsd](https://github.com/amrish1222/traffic_sign_detection/blob/master/results/snippet.gif)


### Running Instructions
1. Move the TSR folder/ dataset folder into the directory Group9 which contains the 2 python scripts-
- Hog_Batch
- SignDetect
2. To run the traffic sign recognition execute the python script- SignDetect.py
3. To train the svm execute the Hog_Batch.py file. This will re-train the SVM and update the required .pkl file.
Please make sure a copy of this made before training if required.
