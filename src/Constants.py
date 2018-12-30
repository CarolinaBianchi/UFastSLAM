
from numpy import zeros, pi
from Sensor import N_MEASUREMENTS, N_SAMPLING
from Vehicle import Vehicle
from numpy import finfo, pi

EPS = finfo(float).eps # Machine epsilon
PI = pi

PATH        = "../victoria_park/"
TIME        = "time.txt"
#TIME
ftime  = open ( PATH + TIME, 'r')
T = [float(line) for line in ftime]
ftime.close()

# H/W parameter(truck).
DT= 0.025; # [s], time interval between control signals
VEHICLE = Vehicle(2.83, 0.76, 0.5, 3.78) # L- H - b- a
#veh= [0 -vehicle.H -vehicle.H; 0 -1 1];

# Control noises
sigmaV= 2 # [m/s]
sigmaG= (6*pi/180) # [rad]
Qe = [[sigmaV**2, 0],[ 0, sigmaG**2]]

# Observation(measurement) noises
sigmaR= 1  # [m]
sigmaB= (3*pi/180) # [rad]
Re = [[sigmaR**2, 0],[0, sigmaB**2]]
perceplimit=30 # [m]

# Resampling criteria
NPARTICLES= 5 # number of particles(samples, hypotheses)
NEFFECTIVE= 0.5*NPARTICLES # minimum number of effective particles

# Data association - innovation gates (Mahalanobis distance)
GATE_REJECT= 5.991     # maximum distance for association
GATE_AUGMENT_NN= 2000  # minimum distance for creation of new feature
GATE_AUGMENT= 100      # minimum distance for creation of new feature (100)

# Parameters related to SUT
dimv= 3
dimQ= 2
dimR= 2
dimf= 2

# Vehicle update
n_aug=dimv+dimf
alpha_aug=0.9; beta_aug=2; kappa_aug=0
lambda_aug = alpha_aug**2 * (n_aug + kappa_aug) - n_aug
lambda_aug=lambda_aug+dimR
wg_aug = zeros((2*n_aug+1)); wc_aug = zeros((2*n_aug+1))
wg_aug[0] = lambda_aug/(n_aug+lambda_aug)
wc_aug[0] = lambda_aug/(n_aug+lambda_aug)+(1-alpha_aug**2+beta_aug)
for i in range(1,(2*n_aug+1)):
    wg_aug[i] = 1/(2*(n_aug+lambda_aug))
    wc_aug[i] = wg_aug[i]


# Vehicle prediction
n_r=dimv+dimQ
alpha_r=0.9
beta_r=2 # optimal for gaussian priors
kappa_r=0
lambda_r = alpha_r**2 * (n_r + kappa_r) - n_r
lambda_r= lambda_r+dimR; # should consider dimension of related terms for obtaining equivalent effect with full augmentation
# (EQ 7)
wg_r = zeros((2*n_r+1)); wc_r = zeros((2*n_r+1))
wg_r[0] = lambda_r/(n_r + lambda_r)                            #weights to compute the mean
wc_r[0] = lambda_r / (n_r+lambda_r) + (1 - alpha_r**2 + beta_r) #weights to compute the covariance
for i in range(1,(2*n_r+1)):
    wg_r[i] = 1/(2*(n_r+lambda_r))
    wc_r[i] = wg_r[i]


# Feature updates (augmented state)
n_f_a= dimf + dimR
alpha_f_a=0.9
beta_f_a=2
kappa_f_a=0
lambda_f_a = alpha_f_a**2 * (n_f_a + kappa_f_a) - n_f_a
wg_f_a = zeros((1,2*n_f_a+1)); wc_f_a = zeros((1,2*n_f_a+1))
wg_f_a[0] = lambda_f_a / (n_f_a + lambda_f_a)
wc_f_a[0] = lambda_f_a / (n_f_a + lambda_f_a) + (1 - alpha_f_a**2 + beta_f_a)
for i in range(1,(2*n_f_a+1)):
    wg_f_a[0,i] = 1/(2*(n_f_a + lambda_f_a))
    wc_f_a[0,i] = wg_f_a[0,i]

# Feature initialization
n_f= dimf
alpha_f=0.9
beta_f=2
kappa_f=0
lambda_f = alpha_f**2 * (n_f + kappa_f) - n_f
wg_f = zeros((1,2*n_f+1)); wc_f = zeros((1,2*n_f+1))
wg_f[0] = lambda_f / (n_f + lambda_f)
wc_f[0] = lambda_f / (n_f + lambda_f) + (1 - alpha_f**2 + beta_f)
for i in range(1,(2*n_f+1)):
    wg_f[0,i] = 1/(2*(n_f+lambda_f))
    wc_f[0,i] = wg_f[0,i]
