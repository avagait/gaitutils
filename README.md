# _gaitutils_ library
Useful code to generate and manipulate data for gait recognition.

In particular, this library offers the following functionality:
* Person detection and tracking in videos.
* Optical flow computation.
* Generation of person-centric optical flow samples. 
Ready to be used with OF-based CNNs for gait recognition.
  

## Demo code at Google Colab

### 1) People detection and tracking

Use the following Google Colab: [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OY-NwJLpgNxRndfgLD0FyML4BYM14niB?usp=sharing)

### 2) Optical flow computation
Use the following Google Colab: [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CPih_tDh4JrkbFBU6kdlg7qVo-SxhF9j?usp=sharing)

### 3) Generate OF-based samples for gait recognition
This step has to be performed after 1 and 2.

You can use the latest section of the previous Colab (in step 2) to generate the actual input samples for the CNN. 
