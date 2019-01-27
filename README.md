# UFastSLAM
Implementation of Unscented Fast SLAM algorithm for Applied Estimation (EL2320) - KTH
This code is the implementation of the algorithm presented by C. Kim in <a href="https://ieeexplore.ieee.org/document/4569861">"Unscented FastSLAM: a Robust and Efficient Solution to the SLAM Problem"</a>. The code is based on the code provided by the <a href="https://github.com/OpenSLAM-org/openslam_ufastslam">author</a>.
<table>
<tbody>
  <tr>
    <td width="50%"><img src="src/results/map_output.gif" "></td>
    <td width="50%"><img src="media/CarolinaBianchi_report-028.jpg"></td>
  </tr>
  </tbody>
</table>
      
## How to run the program
The program has to be run with Python 3 interpreter. The main is located in Uslam.py.
```
  usage: USlam.py [-h] [-p [PLOT]] [-s]
  optional arguments:
  -h, --help  show this help message and exit
  -p [PLOT]   plot option. If the value specified is < = 0, then the nothing
              will be plotted, otherwise the number determines the frequency
              of plot updates (lower = higher frequency).
  -s          if specified, saves all the producted plots in the /output
              folder
```



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
Refers to line 3 of the pseudocode. Implemented in Particle.predictACFRu(self, ctrl)
<table>
  <tbody>
    <tr>
      <td><img src="media/CarolinaBianchi_report-000.png"></td>
      <td><img src="media/CarolinaBianchi_report-001.png"></td>
    </tr>
    <tr>
      <td>Figure 1 – The state of the car is modelled with its mean (red square) and covariance, the dashed pink line represents its 95% confidence interval. Known features are represented in black with the same symbology.</td>
      <td>Figure 2 – We draw a set of sigma points from the vehicle state (red diamonds), each of this corresponds to a specific instance of control noise. Each sigma point is assigned a weight which decreases as the point gets farther from the mean.</td>
    </tr>
    <tr>
      <td><img src="media/CarolinaBianchi_report-002.png"></td>
      <td><img src="media/CarolinaBianchi_report-003.png"></td>
    </tr>
    <tr>
      <td>Figure 3 – The non-linear motion model is applied to each state point using the corresponding control previously drawn. This
produces a set of transformed sigma points (cyan diamonds). The dotted lines represent the path that is point is predicted to follow.</td>
      <td>Figure 4 – The propagated mean and covariance are retrieved from the transformed sigma points. The state covariance
increases, as it has incorporated the control noise. The blue square is the predicted mean, while the blue dashed line is the prediction covariance.</td>
    </tr>
  </tbody>
</table>


## Acknowledgments
Based on the code provided by <a href="https://github.com/OpenSLAM-org/openslam_ufastslam">Kim to Open SLAM community</a>
