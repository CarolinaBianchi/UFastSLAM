# UFastSLAM
Implementation of Unscented Fast SLAM algorithm for Applied Estimation (EL2320) - KTH
This code is the implementation of the algorithm presented by C. Kim in <a href="https://ieeexplore.ieee.org/document/4569861">"Unscented FastSLAM: a Robust and Efficient Solution to the SLAM Problem"</a>. The code is based on the code provided by the <a href="https://github.com/OpenSLAM-org/openslam_ufastslam">author</a>.
![](src/results/map_output.gif)

## Algorithm overview
The high level structure of the algorithm is the following:
``` python
1. while control data is available:
2.     for each particle in the particle set:
3.         predict motion ACFRU using unscented filter
4.         take measurements
5.         for each measurement:
6.             associate measurement to nearest neighbour
7.             if not associated and far enough from nearest neighbour:
8.                 create new feature
9.             else:
10.                discard measurement
11.        Kalman update vehicle state
12.        Kalman update feature state
13.    resample particles
```

## Vehicle state prediction


## Acknowledgments
Based on the code provided by <a href="https://github.com/OpenSLAM-org/openslam_ufastslam">Kim to Open SLAM community</a>
