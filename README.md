**Day 1:** GOAL: Building an object detection app. Ideally want to detect common house-hold items using my own laptop camera. Want to understand principles of spatial learning and computer vision with this project.
* Using YOLOv5 for this project as the model: widely used object detection model. Can identify and classify multiple objects in a **single forward pass** - ideal for real-time detection tasks.
    * Unlike older methods that needed to scan an image multiple times, YOLO only looks at the image once & predicts bounding boxes/class probabilities simultaneously.
    * Special type of CNN built for real-time object detection. 
* YOLO components: 
    (1) CNN feature extractor: converts input image into downsampled feature map with spatial/semantic information.
    (2) Feature fusion (CNN): CNN layers like FPN (feature pyramid network?) to fuse features from different spatial depths (coarse + fine info)
    (3) Final detection layers (Conv layers): at the end, CNNs predict bounding boxes, confidence scores, and class probabilities using convolutions
    * no dense layers or RNNs in architecture -> makes it fast & spatially aware
**Day 2:**
* Github submodule for YOLOV5: clones model repo into folder 
* tracks the submodule by commit hash inside main repo, but not as normal source code
    * keeps it linked to the OG yolov5 repo so we are able to pull upstream changes when needed
    * adv:
        (1) clean separation between your code and model (modular)
        (2) you're not uploading full model history to github
        (3) you can update the model without overwriting your work
        (4) professional approach that is used irl
        (5) model code is read-only
* .pt file stores neural network's parameters / weights, architecture, or both
* Valid pytorch hub repo = having a hubconf.py file 
* cv = computer vision library which handles
    (1) webcam access (videocapture)
    (2) drawing boxes (rectangle)
    (3) displaying window (imshow)
* go through in-line comments in script

WHERE YOU LEFT OFF: creating a UI for your app. Finetuning YOLO with custom images.