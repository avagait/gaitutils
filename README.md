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


## References  

If you either use this code or find useful this repository, please, cite any of the following related works:
```
[A] Francisco M. Castro, Manuel J. Marín-Jiménez, Nicolás Guil, Santiago Lopez Tapia, Nicolas Pérez de la Blanca:
    Evaluation of CNN Architectures for Gait Recognition Based on Optical Flow Maps. BIOSIG 2017: 251-258
[B] Rubén Delgado-Escaño, Francisco M. Castro, Julián Ramos Cózar, Manuel J. Marín-Jiménez, Nicolás Guil:
    MuPeG - The Multiple Person Gait Framework. Sensors 20(5): 1358 (2020)
[C] Francisco M. Castro, Manuel J. Marín-Jiménez, Nicolás Guil, Nicolás Pérez de la Blanca:
    Multimodal feature fusion for CNN-based gait recognition: an empirical comparison. Neural Comput. Appl. 32(17): 14173-14193 (2020)
[D] R. Delgado-Escaño, F. Castro, N. Guil, V. Kalogeiton, M. Marín-Jiménez:  
    Multimodal gait recognition under missing modalities. IEEE ICIP, 2021    
```
