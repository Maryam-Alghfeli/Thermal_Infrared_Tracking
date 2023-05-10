## <div align="center">Real-Time Multi-Object Tracking using YOLOv5 and DeepSORT</div>

Thermal infrared tracking using YOLOv5 and DeepSORT is a powerful computer vision technique that accurately tracks objects in low-light and no-light environments. YOLOv5 is a state-of-the-art object detection algorithm that can accurately detect and classify objects in thermal infrared images. DeepSORT is a sophisticated tracking algorithm that combines appearance features and motion information to assign unique IDs to each tracked object, enabling the tracker to identify and track multiple objects simultaneously accurately.

![FLIR](https://github.com/Maryam-Alghfeli/Thermal_Infrared_Tracking/assets/65627905/ce00de9b-5f46-4117-8e31-86fe694cc5e7) 

By combining the capabilities of YOLOv5 and DeepSORT, it is possible to develop a robust thermal infrared tracking system that can operate in a wide range of challenging environmental conditions. This system can detect and track moving objects and provide real-time location data to the user. Moreover, this system can handle complex situations like object interactions and partial occlusions, making it an effective solution for various applications, including military, security, and surveillance.

Moreover, The Bees Algorithm (BA) is used to find the optimal values of the number of epochs and batches of YOLOv5 to enhance the precision of the used system. The bee consists of the number of epochs, batches, and the precision value obtained from running YOLOv5.



## <div align="center">Credits</div>
The YOLOv5 code used in this project is based on the open-source implementation available on its official GitHub repository. We used this code as a starting point for our implementation and made modifications to suit our specific needs. The original YOLOv5 code can be found at: https://github.com/ultralytics/yolov5. We also incorporated the DeepSORT tracking algorithm available at https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch to enable multi-object tracking."


## <div align="center">Getting Started: Cloning the Repository and Downloading the Environment File</div>

<details open>
<summary>Installation</summary>

Clone repo and install [environment.yml](https://github.com/NoufAlshamsi/Thermal_Infrared_Tracking/blob/main/yolov5/environment.yml).
```bash
git clone https://github.com/NoufAlshamsi/Thermal_Infrared_Tracking
cd yolov5
conda env create -f environment.yml 
```

## <div align="center">Object Detection: Training and Inference with YOLOv5</div>

<details open>

<summary>Training</summary>

The commands below are to train YOLOv5 with FLIR and TII datasets, respectively. Downloading [Models](https://github.com/ultralytics/yolov5/tree/master/models) download automatically from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases). [FLIR dataset](https://www.kaggle.com/datasets/deepnewbie/flir-thermal-images-dataset), and [TII dataset](https://www.herox.com/TIIInfraredTracking/resource/1184) need to download it manually. The results are saved to runs/train.

```bash
python train.py --data FLIR.yaml --epochs 100 --weights '' --cfg yolov5n.yaml  --batch-size 128
                                                                 yolov5s                    64
                                                                 yolov5m                    40
                                                                 yolov5l                    24
                                                                 yolov5x                    16


python train.py --data TII.yaml --epochs 100 --weights '' --cfg yolov5n.yaml  --batch-size 128
                                                                 yolov5s                    64
                                                                 yolov5m                    40
                                                                 yolov5l                    24
                                                                 yolov5x                    16                                   
```
</details>
<details open>

<summary>Inference</summary>

detect.py runs inference on a variety of sources, downloading [Models](https://github.com/ultralytics/yolov5/tree/master/models) download automatically from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases) and saving results to runs/detect.

```bash
python detect.py --weights yolov5s.pt --source 0                               # webcam
                                               img.jpg                         # image
                                               vid.mp4                         # video
                                               screen                          # screenshot
                                               path/                           # directory
                                               list.txt                        # list of images
                                               list.streams                    # list of streams
                                               'path/*.jpg'                    # glob
                                               'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                               'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```
The images are in the [test_images_videos](https://github.com/NoufAlshamsi/Thermal_Infrared_Tracking/tree/main/test_images_videos) folder for running inference on the FLIR or TII datasets.

</details>



## <div align="center">Hyper-Parameter Tuning Using The Bees Algorithm</div>
<details open>

<summary>Training</summary>

The BA implementation is available in the train_BA.py file, which can be run using the command python train_BA.py. In addition, the optimization process results will also be saved in the runs/train directory.

```bash
python train_BA.py --data FLIR.yaml --epochs 100 --weights '' --cfg yolov5n.yaml  --batch-size 128
                                                                    yolov5s                    64
                                                                    yolov5m                    40
                                                                    yolov5l                    24
                                                                    yolov5x                    16


python train_BA.py --data TII.yaml --epochs 100 --weights '' --cfg yolov5n.yaml  --batch-size 128
                                                                    yolov5s                    64
                                                                    yolov5m                    40
                                                                    yolov5l                    24
                                                                    yolov5x                    16  
```
</details>

<details open>
<summary>Inference</summary>

For the argument --weights, we pass the weights produced from the Bees Algorithm, and to run inference on the FLIR or TII datasets, the images are located in the [test_images_videos](https://github.com/NoufAlshamsi/Thermal_Infrared_Tracking/tree/main/test_images_videos) folder, and the results of the inference are saved in the runs/detect.

```bash
python detect.py --weights '' --source 0                               # webcam
                                      img.jpg                         # image
                                      vid.mp4                         # video
                                      screen                          # screenshot
                                      path/                           # directory
                                      list.txt                        # list of images
                                      list.streams                    # list of streams
                                      'path/*.jpg'                    # glob
                                      'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                      'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>


## <div align="center">DeepSORT Tracker</div>

To use the DeepSORT tracker, run the track.py file. The tracker's output will be saved into the runs/track directory. The tracker uses the detections generated by YOLOv5 to track objects and assign unique IDs to each object. 

To run the tracker, specify the YOLOv5 model weights using the --yolo_model argument. You can either use the pre-trained YOLOv5 weights or your own [pre-trained weights](https://drive.google.com/drive/folders/1V8F_2ebTYrZeVUsdBi4mgTXlx4HsU_4c?usp=share_link), depending on whether you want to detect FLIR or TII datasets, respectively, and to run a tracker on the FLIR or TII datasets; the videos are located in the [test_images_videos](https://github.com/NoufAlshamsi/Thermal_Infrared_Tracking/tree/main/test_images_videos) folder. 

```bash
python track.py --yolo_model '' --source img.png

```

</details>


## <div align="center">License</div>

This project is licensed under the MIT License - see the [LICENSE](https;://github.com/ultralytics/yolov5/blob/master/LICENSE) file for details.

## <div align="center">Contact</div>

For any inquiries or support requests, please feel free to contact us at the following email addresses:
- Nouf Alshamsi: nouf.alshamsi@mbzuai.ac.ae
- Maryam Alghfeli: maryam.alghfeli@mbzuai.ac.ae
- Mariam Kashkash: mariam.kashkash@mbzuai.ac.ae

