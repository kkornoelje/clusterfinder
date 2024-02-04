'''
PURPOSE:
Calculate conversion factor to get the CMB temperature
fluctuation from Compton-y map.  The formula looks like this:
delta_T/T = y * (x * (exp(x) + 1)/(exp(x) - 1) - 4)
where x = (h* nu)/(k_B * T_CMB) = nu/56.846 GHz.
(See Shultz and White, Ap.J 586 (2003) 723-730, on the fourth page)                                                                                                                                                                                  
INPUT:
nughz - an array of frequencies in GHz (enter 90GHz as 90)
Tee - an array of temperatures in keV                                                                                                                                                               
OUTPUT: 
inten - an array of intensity in units of W/m**2/sr/Hz
the dimensions of the array will be tee x nue 
(e.g.- if tee = [5,10] and nue = [150, 220, 355], inten will be a fltarr(2,3).                                                                                                                             
Flux (mJy) = inten * 1e26 (Jy/(m^2*Hz)) * Omega_beam(sr) * 1e3 (mJy/Jy) * optical_depth                                                                                                             
i_o - 2*(k T_cmb)^3/(h c)^2 - its common to express inten with this value divided out, so here it is if you want it.                                     
delT = (delI/i_o) * (T_cmb/Omega_beam) * (e^x-1)^2/(x^4*e^x)                                                                                                                                       
elT = inten * optical_depth * T_cmb * (e^x-1)^2 / (x^4*e^x)                                                                                                                                       
KEYWORDS: 
T_o - an optional input of the CMB temperature (assumes 2.725)  

CALLING SEQUENCE:
nu = np.linspace(70,250)
tempkeV = np.asarray([5])
spectrum = fxsz_itoh(nu,tempkeV)

NOTE: May only be valid to ~10 keV
;-----------------------------------------------------------------------------                                                                                                                            
; this program is based off of the Itoh relativistic SZ thermal expansion as given in                                                                   Nozawa, et al., 2000, ApJ, 536, 31.  "Relativistic Corrections IV..."    
'''
import numpy as np
import math

def Ri(THETAE, X):
        A = [    0.0, 4.13674e-03,
        -3.31208e-02,   1.10852e-01,  -8.50340e-01,   9.01794e+00,
         -4.66592e+01,   1.29713e+02,  -2.09147e+02,   1.96762e+02,
         -1.00443e+02,   2.15317e+01,  -4.40180e-01,   3.06556e+00,
         -1.04165e+01,   2.57306e+00,   8.52640e+01,  -3.02747e+02,
          5.40230e+02,  -5.58051e+02,   3.10522e+02,  -6.68969e+01,
         -4.02135e+00,   1.04215e+01,  -5.17200e+01,   1.99559e+02,
         -3.79080e+02,   3.46939e+02,   2.24316e+02,  -1.36551e+03,
          2.34610e+03,  -1.98895e+03,   8.05039e+02,  -1.15856e+02,
         -1.04701e+02,   2.89511e+02,  -1.07083e+03,   1.78548e+03,
         -2.22467e+03,   2.27992e+03,  -1.99835e+03,   5.66340e+02,
         -1.33271e+02,   1.22955e+02,   1.03703e+02,   5.62156e+02,
         -4.18708e+02,   2.25922e+03,  -1.83968e+03,   1.36786e+03,
         -7.92453e+02,   1.97510e+03,  -6.95032e+02,   2.44220e+03,
         -1.23225e+03,  -1.35584e+03,  -1.79573e+03,  -1.89408e+03,
         -1.77153e+03,  -3.27372e+03,   8.54365e+02,  -1.25396e+03,
         -1.51541e+03,  -3.26618e+03,  -2.63084e+03,   2.45043e+03,
          5.10306e+03,   3.58624e+03,   9.51532e+03,   1.91833e+03,
          9.66009e+03,   6.12196e+03,   1.12396e+03,   3.46686e+03,
          4.91340e+03,  -2.76135e+02,  -5.50214e+03,  -7.96578e+03,
         -4.52643e+03,  -1.84257e+04,  -9.27276e+03,  -9.39242e+03,
         -1.34916e+04,  -6.12769e+03,   3.49467e+02,   7.13723e+02,
          7.73758e+03,   5.62142e+03,   4.89986e+03,   3.50884e+03,
          1.86382e+04,   1.71457e+04,   1.45701e+03,  -1.32694e+03,
         -5.84720e+03,  -6.47538e+03,  -9.17737e+03,  -7.39415e+03,
         -2.89347e+03,   1.56557e+03,  -1.52319e+03,  -9.69534e+03,
         -1.26259e+04,   5.42746e+03,   2.19713e+04,   2.26855e+04,
          1.43159e+04,   4.00062e+03,   2.78513e+02,  -1.82119e+03,
         -1.42476e+03,   2.82814e+02,   2.03915e+03,   3.22794e+03,
         -3.47781e+03,  -1.34560e+04,  -1.28873e+04,  -6.66119e+03,
         -1.86024e+03,   2.44108e+03,   3.94107e+03,  -1.63878e+03]
        
        THE     = (THETAE-0.01)*100/4.
        Z       = (X-2.5)/17.5
        
        
        Rr  = (A[1]+A[2]*THE+A[3]*THE**2+A[4]*THE**3+A[5]*THE**4 +A[6]*THE**5+A[7]*THE**6+A[8]*THE**7 +A[9]*THE**8+A[10]*THE**9+A[11]*THE**10) 
        +(A[12]+A[13]*THE+A[14]*THE**2 +A[15]*THE**3+A[16]*THE**4 +A[17]*THE**5+A[18]*THE**6+A[19]*THE**7 +A[20]*THE**8+A[21]*THE**9+A[22]*THE**10)*Z 
        
        +(A[23]+A[24]*THE+A[25]*THE**2+A[26]*THE**3+A[27]*THE**4+A[28]*THE**5+A[29]*THE**6+A[30]*THE**7 +A[31]*THE**8+A[32]*THE**9+A[33]*THE**10)*Z**(2) 
        
        +(A[34]+A[35]*THE+A[36]*THE**2 +A[37]*THE**3+A[38]*THE**4+A[39]*THE**5+A[40]*THE**6+A[41]*THE**7+A[42]*THE**8+A[43]*THE**9+A[44]*THE**10)*Z**(3)
        
        +(A[45]+A[46]*THE+A[47]*THE**2 +A[48]*THE**3+A[49]*THE**4+A[50]*THE**5+A[51]*THE**6+A[52]*THE**7+A[53]*THE**8+A[54]*THE**9+A[55]*THE**10)*Z**(4)
        
        +(A[56]+A[57]*THE+A[58]*THE**2 +A[59]*THE**3+A[60]*THE**4+A[61]*THE**5+A[62]*THE**6+A[63]*THE**7+A[64]*THE**8+A[65]*THE**9+A[66]*THE**10)*Z**(5)
        
        +(A[67]+A[68]*THE+A[69]*THE**2 +A[70]*THE**3+A[71]*THE**4+A[72]*THE**5+A[73]*THE**6+A[74]*THE**7+A[75]*THE**8+A[76]*THE**9+A[77]*THE**10)*Z**(6)
        
        +(A[78]+A[79]*THE+A[80]*THE**2+A[81]*THE**3+A[82]*THE**4+A[83]*THE**5+A[84]*THE**6+A[85]*THE**7+A[86]*THE**8+A[87]*THE**9+A[88]*THE**10) *Z**(7) 
        
        +(A[89]+A[90]*THE+A[91]*THE**2+A[92]*THE**3+A[93]*THE**4+A[94]*THE**5+A[95]*THE**6+A[96]*THE**7+A[97]*THE**8+A[98]*THE**9+A[99]*THE**10)*Z**(8)
        
        +(A[100]+A[101]*THE+A[102]*THE**2 +A[103]*THE**3+A[104]*THE**4+A[105]*THE**5+A[106]*THE**6+A[107]*THE**7+A[108]*THE**8 +A[109]*THE**9 +A[110]*THE**10)*Z**(9)
        
        +(A[111]+A[112]*THE+A[113]*THE**2+A[114]*THE**3+A[115]*THE**4+A[116]*THE**5+A[117]*THE**6+A[118]*THE**7+A[119]*THE**8+A[120]*THE**9
          +A[121]*THE**10) *Z**(10)

        return Rr
    
def fink(thetae, x):
    sh = (math.exp(x / 2) - math.exp(-x / 2)) / 2
    ch = (math.exp(x / 2) + math.exp(-x / 2)) / 2
    cth = ch / sh

    xt = x * cth
    st = x / sh

    y0 = -4.0 + xt

    y1 = -10.0 + 47.0 * xt / 2.0 - 42.0 * (xt ** 2.0) / 5.0 + \
        7.0 * (xt ** 3.0) / 10.0 + \
        (st ** 2.0) * (-21.0 / 5.0 + 7.0 * xt / 5.0)

    y2 = -15.0 / 2 + 1023.0 * xt / 8.0 - 868.0 * (xt ** 2.0) / 5.0 + \
        329.0 * (xt ** 3.0) / 5.0 - 44.0 * (xt ** 4.0) / 5.0 + \
        11.0 * (xt ** 5.0) / 30.0 + (st ** 2.0) * (-434.0 / 5.0 + 658.0 * xt / 5.0 -
                                                 242.0 * (xt ** 2.0) / 5.0 + 143.0 * (xt ** 3.0) / 30.0) + (st ** 4.0) * (
                -44.0 / 5.0 + 187.0 * xt / 60.0)

    y3 = 15.0 / 2 + 2505.0 * xt / 8.0 - 7098.0 * (xt ** 2.0) / 5.0 + \
        14253.0 * (xt ** 3.0) / 10.0 - 18594.0 * (xt ** 4.0) / 35.0 + \
        12059.0 * (xt ** 5.0) / 140.0 - 128.0 * (xt ** 6.0) / 21.0 + \
        16.0 * (xt ** 7.0) / 105.0 + (st ** 2.0) * (-7098.0 / 10.0 + 14253.0 * xt / 5.0 -
                                                   102267.0 * (xt ** 2.0) / 35.0 + 156767.0 * (xt ** 3) / 140.0 -
                                                   1216.0 * (xt ** 4.0) / 7.0 + 64.0 * (xt ** 5.0) / 7.0) + (st ** 4.0) * (
                -18594.0 / 35.0 + 205003.0 * xt / 280.0 - 1920.0 * (xt ** 2.0) / 7.0 + 1024.0 * (xt ** 3.0) / 35.0) + (
                st ** 6.0) * (-544.0 / 21.0 + 992.0 * xt / 105.0)

    y4 = -135.0 / 32.0 + 30375.0 * xt / 128.0 - 62391.0 * (xt ** 2.0) / 10.0 + \
        614727.0 * (xt ** 3.0) / 40.0 - 124389.0 * (xt ** 4.0) / 10.0 + \
        355703.0 * (xt ** 5.0) / 80.0 - 16568.0 * (xt ** 6.0) / 21.0 + \
        7516.0 * (xt ** 7.0) / 105.0 - 22.0 * (xt ** 8.0) / 7.0 + 11.0 * (xt ** 9.0) / 210.0 + (st ** 2.0) * (
                -62391.0 / 20.0 + 614727.0 * xt / 20.0 - 1368279.0 * (xt ** 2.0) / 20.0 + 4624139.0 * (xt ** 3.0) / 80.0 -
                157396.0 * (xt ** 4.0) / 7.0 + 30064.0 * (xt ** 5.0) / 7.0 - 2717.0 * (xt ** 6.0) / 7.0 + 2761.0 * (
                    xt ** 7.0) / 210.0) + (st ** 4.0) * (-124389.0 / 10.0 + 6046951.0 * xt / 160.0 - 248520.0 * (
                xt ** 2.0) / 7.0 + 481024.0 * (xt ** 3.0) / 35.0 - 15972.0 * (xt ** 4.0) / 7.0 + 18689.0 * (
                                                          xt ** 5.0) / 140.0) + (st ** 6.0) * (
                -70414.0 / 21.0 + 465992.0 * xt / 105.0 - 11792.0 * (xt ** 2.0) / 7.0 + 19778.0 * (xt ** 3.0) / 105.0) + (
                st ** 8.0) * (-682.0 / 7.0 + 7601.0 * xt / 210.0)

    f1 = thetae * x * math.exp(x) / (math.exp(x) - 1)

    fnk = f1 * (y0 + thetae * y1 + y2 * thetae ** 2 + y3 * thetae ** 3 + y4 * thetae ** 4)

    return fnk


def sigmoid(t, k=1):
    return 1 / (1 + np.exp(-k * t))


def fxsz_itoh(nughz, tempkeV, tcmb=2.725):
    if isinstance(nughz, list):
        nughz = np.asarray(nughz)
    if isinstance(tempkeV, list):
        tempkeV = np.asarray(tempkeV)
        
    try:
        len(nughz)
    except:
        nughz = np.asarray([nughz])
    
    try:
        len(tempkeV)
    except:
        tempkeV = np.asarray([tempkeV])
    
    h = 6.626076e-34
    k = 1.380658e-23
    m_e = 9.10940e-31
    mc2 = 511.0
    c = 2.997925e8
    
    i_o = 2 * (k * tcmb) ** 3 / h ** 2 / c ** 2
    
    x_all = h * (nughz * 1e9) / k / tcmb
    thetae_all = tempkeV / mc2
    
    ntmp_pts = len(thetae_all)
    nx_pts = len(x_all)
    
    inten = np.zeros((ntmp_pts, nx_pts))
    gx = np.zeros(nx_pts)
    
    for i in range(ntmp_pts):
        thetae = thetae_all[i]
        
        for j in range(nx_pts):
            x = x_all[j]
            
            if thetae < 0.02:
                #Depends on temperature
                #Condition fails around ~10.1 kev
                delI = fink(thetae, x) 
            elif x < 2.5:
                #Depends on frequency
                #Condition fails around ~140GHz
                delI = fink(thetae, x) 
            else:
                transition_midpoint_x = 2.7
                transition_width_x = 0.07  

                # Compute t_x so that it's 0 at transition_midpoint_x
                t_x = (x - transition_midpoint_x) / transition_width_x

                # Apply sigmoid function to get the weight
                #Sigmoid function exists due to discontinuity around ~140GHz that messes with fitting
                #Not sure if this is the best way to handle it
                
                weight_Ri_x = sigmoid(t_x)
                delI = fink(thetae, x) + (weight_Ri_x*Ri(thetae, x))
                
            inten[i, j] = delI * (511.0 / tempkeV[i]) * (np.exp(x) - 1.0) / (x * np.exp(x))
            
            if i == 0:
                gx[j] = i_o * x ** 4 * np.exp(x) / (np.exp(x) - 1.0) ** 2 * (x * (np.exp(x) + 1.0) / (np.exp(x) - 1.0) - 4.0)
                
    return inten

