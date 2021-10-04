# Learning 3D Human Dynamics from Video

This project is a modified version from the original project [Project Page](https://akanazawa.github.io/human_dynamics/). Angjoo Kanazawa*, Jason Zhang*, Panna Felsen*, Jitendra Malik, University of California, Berkeley.

This fork includes the input of 2D files from OpenPose as external system. Make sure you have the AlphaPose and OpenPose installed and update the path to them.

![Teaser Image](resources/overview.jpg)

### Requirements Updated

- Python 3 (tested on version 3.6)
- [TensorFlow](https://www.tensorflow.org/) (tested on version 2.0)
- [PyTorch](https://pytorch.org/) for AlphaPose, PoseFlow, and NMR (tested on
  version 1.1.3)
- [AlphaPose/PoseFlow](https://github.com/akanazawa/AlphaPose)
- [Neural Mesh Renderer](https://github.com/daniilidis-group/neural_renderer)
  for rendering results. See below.
- [CUDA](https://developer.nvidia.com/cuda-downloads) (tested on CUDA 11.2 with GeForce 940MX)
- ffmpeg (tested on version 4.1.3)

There is currently no CPU-only support.

### License
Please note that while our code is under BSD, the SMPL model and datasets we use have their own licenses that must be followed.

### Installation

Tested in Conda environment with python 3.6

Follow all the instruction to install the original project:

#### Install External Dependencies.
Neural Mesh Renderer and AlphaPose for rendering results:
```
cd src/external
sh install_external.sh
```
Install the latest version of [AlphaPose] (https://github.com/MVIG-SJTU/AlphaPose), and [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

### Demo

1. Download the pre-trained models (also available on [Google Drive](https://drive.google.com/file/d/1LlF9Nci8OtkqKfGwHLh7wWx1xRs7oSyF/view)). Place the `models` folder as a top-level
directory.
```
wget http://angjookanazawa.com/cachedir/hmmr/hmmr_models.tar.gz && tar -xf hmmr_models.tar.gz
```
2. Download the `demo_data` videos (also available on [Google Drive](https://drive.google.com/file/d/1ljb8RLJ-PupFlzomhfjpUTzP1Cf6V2no/view)). Place the `demo_data` folder as a top-level
directory.

```
wget http://angjookanazawa.com/cachedir/hmmr/hmmr_demo_data.tar.gz && tar -xf hmmr_demo_data.tar.gz
```
Sample usage:

```
# Run on a single video:
python -m demo_video --vid_path demo_data/penn_action-2278.mp4 --load_path models/hmmr_model.ckpt-1119816

python -m demo_video_openpose --vid_path demo_data/penn_action-2278.mp4 --load_path models/hmmr_model.ckpt-1119816

# If there are multiple people in the video, you can also pass a track index:
python -m demo_video --track_id 1 --vid_path demo_data/insta_variety-tabletennis_43078913_895055920883203_6720141320083472384_n_short.mp4 --load_path models/hmmr_model.ckpt-1119816

# Run on an entire directory of videos:
python -m demo_video --vid_dir demo_data/ --load_path models/hmmr_model.ckpt-1119816
```

This will make a directory `demo_output/<video_name>`, where intermediate
tracking results and our results are saved as video, as well as a pkl file. 
Alternatively you can specify the output directory as well. See `demo_video.py`


### Training code

See [doc/train](doc/train.md).

### Data

#### InstaVariety

![Insta-Variety Teaser](resources/instavariety.gif)


We provided the raw list of videos used for InstaVariety, as well as the
pre-processed files in tfrecords. Please see
[doc/insta_variety.md](doc/insta_variety.md) for more details..

### Citation
If you use this code for your research, please consider citing:
```
@InProceedings{humanMotionKZFM19,
  title={Learning 3D Human Dynamics from Video},
  author = {Angjoo Kanazawa and Jason Y. Zhang and Panna Felsen and Jitendra Malik},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```
