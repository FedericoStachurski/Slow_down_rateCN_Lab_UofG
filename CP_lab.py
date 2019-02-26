#%% import libraries
import matplotlib.pyplot as plt
import numpy as np
from itertools import islice
import operator
import math
import datetime as dt
import textwrap
import astropy
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.metrics import r2_score
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body
from astropy.time import Time
solar_system_ephemeris.set('URL: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de200.bsp')
%matplotlib inline

#%% Constants
a = 3.2 * (10**15) #Magnetic field constant parameter 
ecl = 23.439281 #ecliptic angle
C = 299792.458 #speed of light km/s
R = 1.5*(10**8) # km AU
PSR_EcLon = 83.98316667 #PSR Ecliptic long
PSR_EcLat = -1.29544444 #PSR Ecliptic lat
DM = 56.77118 #Dispersion measure
err_DM = 2.400e-04
DM_const = (4.15*(10**3)) #constant for t_dispersion
Angles = SkyCoord('05 34 31.973 +22 00 52.06',unit=(u.hourangle, u.deg), frame='icrs')  #Right Ascenscion J (SSB) Crab
RAJ = Angles.ra.value # err_RAJ = 5.000e-03  #error RA
DECJ = Angles.dec.value  # err_DECJ = 6.000e-02 #error DEC
print(RAJ,DECJ)
Geo_coord=[52.16361111, 0.3722222] #One Mile telescope coordinates
loc=astropy.coordinates.EarthLocation.from_geodetic(0.3722222, 52.16361111, 22)

#%% path file for data
path="/Users/federicostachurski/Desktop/ASTRO_Lab_PSR-CRAB/"

#%%Read Joerdell Bank observations (1 one more obs)
jb_data_1 = np.loadtxt(path+"JB_data_1992.cpx")
#%%Read Joerdell Bank observations 
jb_data = np.loadtxt(path+"JB_data_1992.cpx")
#%%function for opneing data files
def openfile_data(filename):
    data=np.loadtxt(filename,skiprows=3)
    return data
def openfile_par(filename):
    j=0
    par=np.zeros(3)
    with open(filename) as f:
        for i in islice(f, 3):
            par[j]=i
            j=j+1
    return par

#%% signal data [real imaginary]
data_1=openfile_data(path+"T48669.CPX")
data_2=openfile_data(path+"T48670.CPX")
data_3=openfile_data(path+"T48686.CPX")
data_4=openfile_data(path+"T48687.CPX")
data_5=openfile_data(path+"T48688.CPX")
data_6=openfile_data(path+"T48691.CPX")
data_7=openfile_data(path+"T48693.CPX")
data_8=openfile_data(path+"T48698.CPX")
data_9=openfile_data(path+"T48699.CPX")

#%%parameter data 
# [julian_date, frequency_multiplying_function, time_interval=10.24]
par_1=openfile_par(path+"T48669.CPX")
par_2=openfile_par(path+"T48670.CPX")
par_3=openfile_par(path+"T48686.CPX")
par_4=openfile_par(path+"T48687.CPX")
par_5=openfile_par(path+"T48688.CPX")
par_6=openfile_par(path+"T48691.CPX")
par_7=openfile_par(path+"T48693.CPX")
par_8=openfile_par(path+"T48698.CPX")
par_9=openfile_par(path+"T48699.CPX")

#%%Signal plot(data,time,number, real (0) or imag(1))
def signal_plt(data,par,num,inx):
    T=par[2]
    n=len(data[:,0]) #length data
    t_x=np.arange(0,n*T,T) #time intervals observation
    if inx is 0:
        plt.plot(t_x,data[:,0])
        plt.xlabel('Observational time [s]')
        plt.ylabel('Pulse strength')
        plt.xlim(0,n*T)
        textstr = '\n'.join((
        r"Obs. points = "+str(n),
        r"Time int. = " +str(T)))
        plt.text(1.1, 0.7, textstr, fontsize=15,
            verticalalignment='top')
        plt.title('data_'+num)
    elif inx is 1:
        plt.plot(t_x,data[:,1],'c')
        plt.xlabel('Observational time [s]')
        plt.ylabel('Pulse strength')
        plt.xlim(0, 40000)
        textstr = '\n'.join((
        r"Obs. points = "+str(n),
        r"Time int. = " +str(T)))
        plt.text(1.1, 0.7, textstr, fontsize=15,
            verticalalignment='top')
        plt.title('data_'+num)
    else: 
        print('No Inx was submitted, signal_plt requires 4 arg, or number inx not 0 or 1')
        return
    plt.show()
    return n,T
#%% plot signal
Signal_P1=signal_plt(data_1,par_1,'1',1)
Signal_P2=signal_plt(data_2,par_2,'2',1)
Signal_P3=signal_plt(data_3,par_3,'3',0)
Signal_P4=signal_plt(data_4,par_4,'4',0)
Signal_P5=signal_plt(data_5,par_5,'5',0)
Signal_P6=signal_plt(data_6,par_6,'6',0)
Signal_P7=signal_plt(data_7,par_7,'7',0)
Signal_P8=signal_plt(data_8,par_8,'8',0)
Signal_P9=signal_plt(data_9,par_9,'9',0)


#%%FFT (function), takes data col 1 and 2, makes complex num with 
#data, runs complex data in FFT, returns FFT, plts FFT of Signals

def FastFT(data,par,inx):
    T=par[2]
    Jdate=par[0]
    n=len(data[:,0])
    f=abs(np.fft.fftfreq(n,T)-par[1])
    tot_data=np.zeros(n)+complex(0,0)
    i=0
    while i<n:
        tot_data[i]=np.complex(data[i,0],data[i,1])
        i=i+1
    FFT=abs(np.fft.fft(tot_data))*abs(np.fft.fft(tot_data))/n
    if inx is 0:
        plt.plot(f,FFT.real,'r')
        plt.ylabel('Signal Real FFT')
        plt.xlabel('Frequency [Hz]')
        plt.title('Date ' +str(Jdate))
        plt.show()
    elif inx is 1:
        plt.plot(f,FFT.imag,'b')
        plt.ylabel('Signal Imag FFT')
        plt.xlabel('Frequency [Hz]')
        plt.title('Date ' +str(Jdate))
        plt.show()
    else: 
        print('No Inx was submitted, FastFT requires 4 arg, or number inx not 0 or 1')
        return
    return FFT,f

Signal_F1=FastFT(data_1,par_1,0)
Signal_F2=FastFT(data_2,par_2,0)
Signal_F3=FastFT(data_3,par_3,0)
Signal_F4=FastFT(data_4,par_4,0)
Signal_F5=FastFT(data_5,par_5,0)
Signal_F6=FastFT(data_6,par_6,0)
Signal_F7=FastFT(data_7,par_7,0)
Signal_F8=FastFT(data_8,par_8,0)
Signal_F9=FastFT(data_9,par_9,0)

#%% Max_Inx find max value of FFT, return index
def Max_Inx(data):
    max = 0
    max_inx=0
    i=0
    n=len(data)
    data_abs=abs(data)
    while i<n :
        if np.real(data_abs[i])>max :
            max=np.real(data_abs[i])
            max_inx=i
        i=i+1
    return max_inx

#finds freq associated w/ index (real)
def findFREQ(data,inx):
    if inx is 0:
        Freq=data[1][Max_Inx(np.real(data[0]))]
        print(Freq,(np.real(data[0][Max_Inx(np.real(data[0]))])))
    elif inx is 1:
        Freq=data[1][Max_Inx(np.imag(data[0]))]
        print(Freq,(np.imag(data[0][Max_Inx(np.imag(data[0]))])))
    else:
        print('please input 2 arg, (inx (real 0 or imag 1))')
    return Freq

# Freq_1=findFREQ(Signal_F1,0)
# Freq_2=findFREQ(Signal_F2,0)
Freq_3=findFREQ(Signal_F3,0)
Freq_4=findFREQ(Signal_F4,0)
Freq_5=findFREQ(Signal_F5,0)
Freq_6=findFREQ(Signal_F6,0)
Freq_7=findFREQ(Signal_F7,0)
Freq_8=findFREQ(Signal_F8,0)
Freq_9=findFREQ(Signal_F9,0)

#%% Frequency array ,Time array and average Frequency
i=0
l=7
Freq_array=np.zeros(l)
Jdate_array=np.zeros(l)
while i<l:
    Freq_array[i]=vars()['Freq_'+str(i+3)]
    Jdate_array[i]=vars()['par_'+str(i+3)][0]
    i=i+1
# Avg_Freq=np.mean(Freq_array)
# print(Avg_Freq)
print(Jdate_array)

#%% Change JDate in Standard Date
def jd_to_date(jd):
    jd = jd + 0.5
    F, I = math.modf(jd)
    I = int(I)
    A = math.trunc((I - 1867216.25)/36524.25)
    if I > 2299160:
        B = I + 1 + A - math.trunc(A / 4.)
    else:
        B = I
    C = B + 1524
    D = math.trunc((C - 122.1) / 365.25)
    E = math.trunc(365.25 * D)
    G = math.trunc((C - E) / 30.6001)
    day = C - E + F - math.trunc(30.6001 * G)
    if G < 13.5:
        month = G - 1
    else:
        month = G - 13
    if month > 2.5:
        year = D - 4716
    else:
        year = D - 4715
    frac_day = day - int(day)
    hours = frac_day * 24.
    hours, hour = math.modf(hours)
    mins = hours * 60.
    mins, min = math.modf(mins)
    secs = mins * 60.
    secs, sec = math.modf(secs)
    micro = round(secs * 1.e6)
    return year, month, int(day), int(hour), int(min), int(sec)


#%%true JD to Date and find positions 

jd_delay = np.asarray(t_delay + 2400000)
date_delay = list(map(jd_to_date,jd_delay))
print(date_delay )
#%%
k=0
while k<7:
    if date_delay[k][-1] >= 30 : 
        date_delay[k][-2] >= date_delay[k][-2]+1

    del date_delay[k][-1]

print(date_delay)

#%%
k=0
pos_vector_delay=np.zeros(7)
while k<=6:
    j=0
    while j<len(date_eph_vector):
        if date_eph_vector[j] == date_delay[]:
            pos_vector_delay[k]=j
        j=j+1
    k=k+1 
#%%
date_1=list(jd_to_date(par_1[0]))
del date_1[-1]
date_2=list(jd_to_date(par_2[0]))
del date_2[-1]
date_3=list(jd_to_date(par_3[0]))
del date_3[-1]
date_4=list(jd_to_date(par_4[0]))
del date_4[-1]
date_5=list(jd_to_date(par_5[0]))
del date_5[-1]
date_6=list(jd_to_date(par_6[0]))
del date_6[-1]
date_7=list(jd_to_date(par_7[0]))
del date_7[-1]
date_8=list(jd_to_date(par_8[0]))
del date_8[-1]
date_9=list(jd_to_date(par_9[0]))
del date_9[-1]
#%%
print(date_1)
print(date_2)
print(date_3)
print(date_4)
print(date_5)
print(date_6)
print(date_7)
print(date_8)
print(date_9)

#%% standard date to seconds from epoch of first observation (data 3)
def date_to_sec(date):
    per_year = 3600*24*365.25
    per_day = 3600 *24 
    per_hour = 3600
    per_minute = 60 
    seconds = (date[0]*per_year)+(date[1]*per_day)+(date[2]*per_hour)+(date[3]*per_minute)+date[4]
    return seconds 
print (date_to_sec(date_2))
i=0
l=7
date_sec=np.zeros(l)
while i<l:
    date_sec[i]=date_to_sec(vars()['date_'+str(i+3)])
    i=i+1
print(date_sec)


#%% change mjd into jd
def jd_to_mjd(jd):
     return jd - 2400000.5


#%% read vector ephemeris (VECTOR) ORBIT
#read Earth-Sun Ephemeris from 1992/3/4 - 1992/3/17
ephe_v_vector=np.genfromtxt(path+"horizons_results_vector_2.txt", delimiter=',')

v_x_ephe_vector = ephe_v_vector[:,5]
v_y_ephe_vector = ephe_v_vector[:,6]
v_z_ephe_vector = ephe_v_vector[:,7]
jd_vector = ephe_v_vector[:,0]
r_x_ephe_vector = ephe_v_vector[:,2]
r_y_ephe_vector = ephe_v_vector[:,3]
r_z_ephe_vector = ephe_v_vector[:,4]

#%% read vector ephemeris (VECTOR) ROTATIONAL
time=np.array(Jdate_array)
t=astropy.time.Time(time, format='jd')
c=astropy.coordinates.get_body('Earth', t , loc)
icrs = c.transform_to('icrs')
rotational_vel=icrs.obsgeovel
vel_rot=rotational_vel.get_xyz()
print(vel_rot.value)
vel_rot_x = vel_rot.value[0]
vel_rot_y = vel_rot.value[1]
vel_rot_z = vel_rot.value[2]

#%% fix 4th elemtn of vectors 
vel_rot_x[3] = np.array(vel_rot_x[2] + vel_rot_x[4])/2
vel_rot_y[3] = np.array(vel_rot_y[2] + vel_rot_y[4])/2
vel_rot_z[3] = np.array(vel_rot_z[2] + vel_rot_z[4])/2
#print(vel_rot_x,vel_rot_y,vel_rot_z)

#%% convert into date (VECTOR) orb
date_eph_vector=list(map(jd_to_date,jd_vector))
date_eph_vector=list(map(list, date_eph_vector))

#%% print(date_eph_vector) orb
i=0
while i<len(date_eph_vector):
    if date_eph_vector[i][5] == 59 :
        date_eph_vector[i][4] = date_eph_vector[i][4] + 1
    del date_eph_vector[i][-1]
    i=i+1
print(date_eph_vector)

#%% check position (VECTOR) orb
k=0
pos_vector=np.zeros(7)
while k<=6:
    j=0
    while j<len(date_eph_vector):
        if date_eph_vector[j] == vars()['date_'+str(k+3)]:
            pos_vector[k]=j
        j=j+1
    k=k+1    

pos_vector=list(map(int,pos_vector))
print(pos_vector)

#%% vectors at dates Vx, Vy, Vz and ditances x, y, z orb
i=0
v_x_orb = np.zeros(7)
v_y_orb = np.zeros(7)
v_z_orb = np.zeros(7)
r_x_orb = np.zeros(7)
r_y_orb = np.zeros(7)
r_z_orb = np.zeros(7)
while i<7:
    v_x_orb[i] = v_x_ephe_vector[pos_vector[i]]
    v_y_orb[i] = v_y_ephe_vector[pos_vector[i]]
    v_z_orb[i] = v_z_ephe_vector[pos_vector[i]]
    r_x_orb[i] = r_x_ephe_vector[pos_vector[i]]
    r_y_orb[i] = r_y_ephe_vector[pos_vector[i]]
    r_z_orb[i] = r_z_ephe_vector[pos_vector[i]]

    i=i+1

print(v_y_orb)
print(r_y_orb)

#%% Total vector (v_orb + v_rot)
tot_v_x = v_x_orb + (vel_rot_x*(10**(-3)))
tot_v_y = v_y_orb + (vel_rot_y*(10**(-3)))
tot_v_z = v_z_orb + (vel_rot_z*(10**(-3)))
print(np.sqrt((tot_v_x**2)+(tot_v_y**2)+(tot_v_z**2)))
print(tot_v_z)

#%% unit vector position Crab Nebula 52.16361111, 0.3722222
x_unit = np.cos(PSR_EcLat*0.017453292519)*np.cos((PSR_EcLon)*0.017453292519)
y_unit = np.cos(PSR_EcLat*0.017453292519)*np.sin((PSR_EcLon)*0.017453292519)
z_unit = np.sin(PSR_EcLat*0.017453292519)
unit_vector=[x_unit, y_unit, z_unit]
print(unit_vector)

#%% error on unit vector
sine = [np.sin(PSR_EcLat*0.017453292519), np.sin((PSR_EcLon)*0.017453292519)]
cosine = [np.cos(PSR_EcLat*0.017453292519), np.cos((PSR_EcLon)*0.017453292519)]
tan = [np.tan(PSR_EcLat*0.017453292519), np.cos((PSR_EcLon)*0.017453292519)]
err_Beta = 6.000*(10**(-2))*0.017453292519
err_Lambda = 5.000*(10**(-3))*0.017453292519
error_x = np.sqrt(((err_Betasine[0])**2)+(((1-err_Lambda**2)/cosine[1])**2))
error_y = np.sqrt(((err_Beta/sine[0])**2)+((err_Lambda/sine[1])**2))
error_z = np.sqrt(((err_Lambda/sine[0])**2))
tot_err = np.sqrt((error_x**2)+(error_y**2)+(error_z**2)+((1/(4096*10.24))**2))
print(tot_err)
#%% dot product for each date (velocity)
i=0
vel_for_doppler=np.zeros(7)
while i<7:
    vel_for_doppler[i]=np.dot([tot_v_x[i], tot_v_y[i], tot_v_z[i]],unit_vector)
    i=i+1
print(1/(1+(vel_for_doppler/C)))
print(vel_for_doppler)

#%% dot product for each date (roemer delay)
i=0
delay_R = np.zeros(7)
while i<7:
    delay_R[i]=(np.dot([r_x_orb[i], r_y_orb[i], r_z_orb[i]],unit_vector))/C
    i=i+1
print(delay_R/(86400))

#%%Doppler (corrected V_orb V_rot)
def doppler(freq,velocity):
    factor = (1 - ((velocity)/C))
    corr_freq = freq * factor
    print((velocity)/C)
    #print(-Freq_array*(velocity)/C)
    return corr_freq

doppler_freq = doppler(Freq_array,vel_for_doppler)

#%% time delay ( )
times = [4096*10.24/(3600*48), 4096*10.24/(3600*48), 4096*10.24/(3600*48), 4096*10.24/(3600*48), 3258*10.24/(3600*48), 4096*10.24/(3600*48), 4096*10.24/(3600*48)]
t_delay = jd_to_mjd(Jdate_array) + (delay_R/86400) + times
#delta_freq_1 = (slope * (vel_for_doppler/C))*(51544-jd_to_mjd(Jdate_array))
print((delay_R/86400))

#%% plot slowdown frequency w/ doppler shift
plt.plot(t_delay,doppler_freq,'ro-')
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
ax.errorbar(t_delay, doppler_freq, (1/(4096*10.24)),fmt='ro',ls='none')
plt.grid(True)
slope, intercept = np.polyfit(jd_to_mjd(Jdate_array), doppler_freq, 1)
slope_jb, intercept_jb = np.polyfit(jb_data[:,1], jb_data[:,0], 1)
plt.plot(t_delay,(slope*t_delay)+intercept, 'g')
plt.plot(jb_data[:,1],jb_data[:,0],'bo-')
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
textstr = '\n'.join((
r"Slope JB = "+str(float(slope_jb)),
r"Intercept JB = "+str(float(intercept_jb)),
r"Slope Data (corr)= " +str(float(slope)),
r"Intercept Data (corr) = " +str(float(intercept))))
plt.text(0.7,0.4, textstr,fontsize=11,horizontalalignment='left',transform=ax.transAxes,bbox=dict(facecolor='white', alpha=1))
plt.ylabel('Frequency [Hz]')
plt.xlabel('MJD')
plt.title('Frquency Slow down (Doppler corrected)')

#%% plot residuals doppler vs JB
JB_slowdown = (float(slope_jb)*t_delay)+float(intercept_jb)
plt.plot(jd_to_mjd(Jdate_array),(JB_slowdown-(doppler_freq)),'ko')
plt.errorbar(jd_to_mjd(Jdate_array), (JB_slowdown-(doppler_freq)), error ,capsize=7,ecolor='k',ls='none')
plt.ylabel('Frequency Residuals [Hz]')
plt.xlabel('MJD')
plt.grid(True)
#print()

#%% plot JB vs doppler shift frequency zoomed in
slope, intercept = np.polyfit(t_delay, doppler_freq, 1)
coefficient_of_dermination = round(r2_score(doppler_freq, float(slope)*t_delay+float(intercept)),7)
print(slope, intercept,coefficient_of_dermination)
error = [1/(4096*10.24), 1/(4096*10.24), 1/(4096*10.24), 1/(4096*10.24), 1/(3258*10.24), 1/(4096*10.24),1/(4096*10.24)]
fig = plt.figure()
plt.plot(t_delay,doppler_freq,'ko')
A, = plt.plot(t_delay,float(slope)*t_delay+float(intercept),'r-',label='Best Fit')
plt.grid(True)
textstr = "$R^2$ = "+str(float(coefficient_of_dermination))
plt.text(1,1, textstr,fontsize=16,transform=ax.transAxes, alpha=1)
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
AAA = plt.errorbar(t_delay, doppler_freq, error ,capsize=7,ecolor='k',ls='none',fmt="ko",label='Experimental Data')
AA, = plt.plot(t_delay,JB_slowdown,'b-',label = 'Jodrell Bank')
AAAA, = plt.plot([],[],' ',label='$R^2$ = Coeff. of Determination')
plt.legend(handles=[A, AA, AAA, AAAA],fontsize=10)
plt.ylabel('Frequency [Hz]')
plt.xlabel('MJD')
fig.savefig(path+'/data/Plots/Slowdown_RatevsJodrell.png', dpi=300, bbox_inches = "tight")

#%% plot  doppler starting from frist observation 
error = [1/(4096*10.24), 1/(4096*10.24), 1/(4096*10.24), 1/(4096*10.24), 1/(3258*10.24), 1/(4096*10.24),1/(4096*10.24)]
plt.plot((t_delay-t_delay[0])*86400,doppler_freq,'ro-')

slope_1, intercept_1 = np.polyfit((t_delay-t_delay[0])*86400, doppler_freq, 1)
plt.grid(True)
textstr = '\n'.join((
r"Slope Data (corr)= " +str(float(slope_1)),
r"Intercept Data (corr) = " +str(float(intercept_1))))
plt.text(0.7,0.4, textstr,fontsize=11,horizontalalignment='left',transform=ax.transAxes,bbox=dict(facecolor='white', alpha=1))
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
plt.errorbar((t_delay-t_delay[0])*86400, doppler_freq, error ,capsize=7,ecolor='r')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Seconds')

#%% Pulsar timing corrections
#read Earth-Sun Ephemeris from 1992/3/4 - 1992/3/17
def read_ephe(c):
    ephe=open(path+"horizons_result_3.txt", "r")
    lines = ephe.readlines()
    column = []
    i = 0
    for line in lines:
        newArr = (line.split())
        for x in newArr:
            if len(x) < 3:
                newArr.remove(x)
        column.append(float(newArr[c]))
    return column
    ephe.close()

jdate_ephe=read_ephe(0)
RA_ephe=read_ephe(1)
DEC_ephe=read_ephe(2)
Vobs_ephe=read_ephe(4)
EcLON_ephe=read_ephe(5)
EcLAT_ephe=read_ephe(6)

#%%convert Jdate to Standard Date 
date_eph=list(map(jd_to_date,jdate_ephe))
date_eph=list(map(list, date_eph))
print(date_eph[-1])

#%% remove seconds
i=0
while i<len(date_eph):
    if date_eph[i][5] == 59 :
        date_eph[i][4] = date_eph[i][4] + 1
    del date_eph[i][-1]
    i=i+1
print(date_eph[770])

#%%check position of dates in ephemeris
k=0
pos=np.zeros(7)
while k<=6:
    j=0
    while j<len(date_eph):
        if date_eph[j] == vars()['date_'+str(k+3)]:
            pos[k]=j
        j=j+1
    k=k+1    

pos=list(map(int,pos))
print(pos)

#%% get values corresponding to dates
Vobs_ephe_3 = Vobs_ephe[pos[0]]
EcLON_ephe_3 = EcLON_ephe[pos[0]]
Vobs_ephe_4 = Vobs_ephe[pos[1]]
EcLON_ephe_4 = EcLON_ephe[pos[1]]
Vobs_ephe_5 = Vobs_ephe[pos[2]]
EcLON_ephe_5 = EcLON_ephe[pos[2]]
Vobs_ephe_6 = Vobs_ephe[pos[3]]
EcLON_ephe_6 = EcLON_ephe[pos[3]]
Vobs_ephe_7 = Vobs_ephe[pos[4]]
EcLON_ephe_7 = EcLON_ephe[pos[4]]
Vobs_ephe_8 = Vobs_ephe[pos[5]]
EcLON_ephe_8 = EcLON_ephe[pos[5]]
Vobs_ephe_9 = Vobs_ephe[pos[6]]
EcLON_ephe_9 = EcLON_ephe[pos[6]]

print(np.cos(EcLON_ephe_3*0.017453292519),Vobs_ephe_3)

#%% TOA delays (DM and Roemer) (PROBLEM)
def time_delays(freq, angle): 
    t_r = (R/c)*np.cos((np.asarray(angle)-PSR_EcLon)*0.017453292519)*np.cos(PSR_EcLat*0.017453292519)
    t_dm= DM_const*(1/(freq**2))*DM
    f_err=[1/t_r, 1/t_dm]
    return f_err #returns frequency errors due to Roemer and DM

f_delay_3 = time_delays(corr_freq_arr[0],EcLON_ephe_3)
f_delay_4 = time_delays(corr_freq_arr[1],EcLON_ephe_4)
f_delay_5 = time_delays(corr_freq_arr[2],EcLON_ephe_5)
f_delay_6 = time_delays(corr_freq_arr[3],EcLON_ephe_6)
f_delay_7 = time_delays(corr_freq_arr[4],EcLON_ephe_7)
f_delay_8 = time_delays(corr_freq_arr[5],EcLON_ephe_8)
f_delay_9 = time_delays(corr_freq_arr[6],EcLON_ephe_9)

print(f_delay_3,f_delay_4,f_delay_5,f_delay_6,f_delay_7,f_delay_8,f_delay_9)


#%% Chi_squared function
def chi_squared(array,funct):
    std = np.std(array)
    i = 0
    chi_squared=np.zeros(len(array))
    while i<len(array) :
        chi_squared[i] = ((funct[i] - array[i] )**2) / std
        i=i+1
    chi_squared=np.sum(chi_squared)
    return chi_squared

linaer_fit = (slope*jd_to_mjd(Jdate_array))+intercept
CHI_SQ=chi_squared(corr_freq_arr,linaer_fit )
print(CHI_SQ)



















#%% likelyhood functions
error = [1/(4096*10.24), 1/(4096*10.24), 1/(4096*10.24), 0.5/(4096*10.24), 1/(3258*10.24), 1/(4096*10.24),1/(4096*10.24)]
err = error
Freq_corr = doppler_freq
epoch = 3 # the index of the epoch of the solution
t = (jd_to_mjd(Jdate_array) - jd_to_mjd(Jdate_array[epoch]))*24*3600
best_f= np.polyfit(t-t[epoch],Freq_corr,1)
f_funct = np.poly1d(best_f)
# print(best_f)
fdot=best_f[0]
f_mu=best_f[1]
print(fdot)
print(f_mu)
print(Freq_corr)
print(t)
print(best_f)
def logL1(fdot,f): # the log likelihood as a function of fdot and f
    sum = 0
    for i in range(0,7):
        sum += (Freq_corr[i] - (fdot*t[i] + f))**2/err[i]**2
    return -sum/2
def logL2(tau,B): # the log likelihood as a function of B and tau
# using f = a/B*(2*tau)**(-1/2), fdot = -a/B*(2*tau)**(-3/2) where a = 3.2e-15 and B is in tesla
    sum = 0
    for i in range(0,7):
        sum += (Freq_corr[i] - (-a/B*(2*tau)**(-3/2)*t[i] + a/B*(2*tau)**(-1/2)))**2/err[i]**2
    return -sum/2
def logL3(tau,Q): # the log likelihood as a function of tau and Q=B**2*tau = a/2/f**2 (see later)
    sum = 0
    for i in range(0,7):
        sum += (Freq_corr[i] - (-1/2/tau*(a/2/Q)**0.5 *t[i] + (a/2/Q)**0.5))**2/err[i]**2
    return -sum/2

def logL4(tau,f): # the log likelihood as a function of tau and f (see later)
    sum = 0
    for i in range(0,7):
        sum += (Freq_corr[i] - (-f/2/tau*t[i] + f))**2/err[i]**2
    return -sum/2

def logL5(B,f): # the log likelihood as a function of B and f (see later)
    sum = 0
    for i in range(0,7):
        sum += (Freq_corr[i] - (-((B/a)**2)*(f**3)*t[i] + f))**2/err[i]**2
    return -sum/2   

def logL6(B,Q): # the log likelihood as a function of B and Q (see later)
    sum = 0
    f = (a/2/Q)**(0.5)
    fdot = (B**2)*((2*Q)**(-3/2))*(a**(-1/2))
    for i in range(0,7):
        sum += (Freq_corr[i] - (-fdot*t[i] + f))**2/err[i]**2
    return -sum/2   

def logL7(B,fdot): # the log likelihood as a function of B and fdot (see later)
    sum = 0
    f = ((a/B)**(2)*abs(fdot))**(1/3)
    for i in range(0,7):
        sum += (Freq_corr[i] - (fdot*t[i] + f))**2/err[i]**2
        #print(f)
    return -sum/2  

def logL8(tau,fdot): # the log likelihood as a function of tau and fdot (see later)
    sum = 0
    for i in range(0,7):
        sum += (Freq_corr[i] - (fdot*t[i] + 2*abs(fdot)*tau))**2/err[i]**2
    return -sum/2  

#%% plot likelyhood (f fdot)
#frequency search box size
f_width = err[0]*2
#fdot search box size
fdot_width = err[0]*2/t[-1]
# box centre
f_cent = f_funct(0)
fdot_cent = (f_funct(500)-f_funct(-500))/(1000)
boxres=200 #box resolution
f_vals = f_cent + np.linspace(-f_width , f_width, boxres)
fdot_vals = fdot_cent + np.linspace(-fdot_width, fdot_width, boxres)
X,Y = np.meshgrid(fdot_vals, f_vals)
Z = logL1(X, Y)
prob = np.exp(Z-Z.max())
fig, ax = plt.subplots()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
p = ax.pcolor(X, Y, prob, cmap='jet', vmin=prob.min(), vmax=prob.max())
cm = ax.contour(X, Y, prob, (0.1, 0.5, 0.9),alpha=1,linewidths=2)
ax.clabel(cm, inline=True)
cb = fig.colorbar(p)
plt.xlabel('$\dot{f}$ (Hz/s)')
plt.ylabel('frequency (Hz)')
plt.show()
#fig.savefig(path+'/data/Plots/f_fdot.png', dpi=300, bbox_inches = "tight")

# %%plot f Marginals
pfreq = np.sum(prob,1)
fig = plt.figure()
plt.plot(Y[:,0],pfreq,'k')
plt.grid(True)
plt.xlabel('frequency (Hz)')
plt.ylabel('$\propto p(f|D)$')
ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_useOffset(False)
plt.show()
fig.savefig(path+'/data/Plots/f_marginal.png',dpi=300, bbox_inches = "tight")
# %%plot fdot  Marginals
pfdot = np.sum(prob,0)
fig = plt.figure()
plt.plot(X[0,:],pfdot,'k')
plt.grid(True)
plt.xlabel('$\dot{f}$ (Hz/s)')
plt.ylabel('$\propto p(\dot{f}|D)$')
plt.show()
#fig.savefig(path+'/data/Plots/fdot_marginal.png', dpi=300, bbox_inches = "tight")

f_dot_FWHM = FullWHM(pfdot,X[0,:])
print(f_dot_FWHM)

#%% 3D plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.get_yaxis().get_major_formatter().set_useOffset(False)
ax.plot_surface(X, Y, prob, alpha=0.3)
cset = ax.contour(X, Y, prob, zdir='z', cmap='jet')
ax.view_init(30, 30)
plt.show()

#%% try the B-tau plane
# box centre
B_cent = a*(-fdot_cent/f_cent**3)**0.5
tau_cent = -f_cent/2/fdot_cent
#B search box size
B_width = 0.1*B_cent
#tau search box size
tau_width = 0.1*tau_cent
boxres = 100 # number of pixels in each axis
B_vals = B_cent + np.linspace(-B_width , B_width, boxres)
tau_vals = tau_cent + np.linspace(-tau_width, tau_width, boxres)
X,Y = np.meshgrid(tau_vals, B_vals)
Z = logL2(X, Y)
#prob = np.exp( (Z-Z.max()))
prob = np.exp( (Z-Z.max())/1e9) # soften the prob by a large factor to let it show better
fig, ax = plt.subplots()
p = ax.pcolor(X, Y, prob, cmap='jet', vmin=prob.min(), vmax=prob.max())
cb = fig.colorbar(p)
cb.ax.set_ylabel('Prior prbability', rotation=270,labelpad=15)
p = ax.pcolor(X, Y, prob, cmap='jet', vmin=prob.min(), vmax=prob.max())
cm = ax.contour(X, Y, prob, np.arange(0,1,0.2),alpha=1,linewidths=2)
ax.clabel(cm, inline=True)
plt.xlabel('characteristic age, $\\tau $, (s)')
plt.ylabel('magnetic field, $B$, (T)')
plt.show()
fig.savefig(path+'/data/Plots/B_tau.png', dpi=300, bbox_inches = "tight")


#%%Try the Q-tau plane, where Q = B**2 * tau =  a/2/f**2
# box centre
Q_cent = a/2/f_cent**2
tau_cent = -f_cent/2/fdot_cent
#B search box size
Q_width = 3e-6*Q_cent
#tau search box size
tau_width = 0.1*tau_cent
boxres = 200 # number of pixels in each axis
Q_vals = Q_cent + np.linspace(-Q_width , Q_width, boxres)
tau_vals = tau_cent + np.linspace(-tau_width, tau_width, boxres)
X,Y = np.meshgrid(tau_vals, Q_vals)
Z = logL3(X, Y)
prob = np.exp(Z-Z.max())
fig, ax = plt.subplots()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
p = ax.pcolor(X, Y, prob, cmap='jet', vmin=prob.min(), vmax=prob.max())
cb = fig.colorbar(p)
cb.ax.set_ylabel('Prior prbability', rotation=270,labelpad=15)
p = ax.pcolor(X, Y, prob, cmap='jet', vmin=prob.min(), vmax=prob.max())
cm = ax.contour(X, Y, prob, (0.1, 0.2, 0.4, 0.66, 0.9),alpha=1,linewidths=2)
ax.clabel(cm, inline=True)
plt.xlabel('characteristic age, $\\tau $, (s)')
plt.ylabel('$Q$')
plt.show()
#fig.savefig(path+'/data/Plots/Q_tau.png', dpi=300, bbox_inches = "tight")
#%% Try f-tau plane
# box centre
tau_cent = -f_cent/2/fdot_cent
f_cent = f_funct(0)
#frequency search box size
f_width = err[0]*2
#tau search box size
tau_width = 0.1*tau_cent
boxres = 200 # number of pixels in each axis
f_vals = f_cent + np.linspace(-f_width , f_width, boxres)
tau_vals = tau_cent + np.linspace(-tau_width, tau_width, boxres)
X,Y = np.meshgrid(tau_vals, f_vals)
Z = logL4(X, Y)
prob_Qt = np.exp(Z-Z.max())
fig, ax = plt.subplots()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
p = ax.pcolor(X, Y, prob, cmap='jet', vmin=prob.min(), vmax=prob.max())
cb = fig.colorbar(p)
cb.ax.set_ylabel('Prior prbability', rotation=270,labelpad=15)
p = ax.pcolor(X, Y, prob, cmap='jet', vmin=prob.min(), vmax=prob.max())
cm = ax.contour(X, Y, prob, ( 0.2, 0.5, 0.9),alpha=1,linewidths=2)
ax.clabel(cm, inline=True)
plt.xlabel('characteristic age, $\\tau $, (s)')
plt.ylabel('frequency (Hz)')
plt.show()
#fig.savefig(path+'/data/Plots/f_tau.png', dpi=300, bbox_inches = "tight")
# unnormalised marginal for tau:
ptau = np.sum(prob,1)
fig, ax = plt.subplots()
plt.grid(True)
plt.plot(X[1,:],ptau,'k')
plt.xlabel('characteristic age, $\\tau $, (s)')
plt.ylabel('$\\propto p(\\tau | D)$')
plt.show()
fig.savefig(path+'/data/Plots/tau_f_marginal.png', dpi=300, bbox_inches = "tight")
#%% normalised marginal for tau:
ptau2 = np.sum(prob_Qt,1)
dtau = X[1,1]-X[1,0] # the width of a tau bin
fig, ax = plt.subplots()
tauf, = plt.plot(X[1,:],ptau2/np.sum(ptau2)/dtau, 'b',label='uniform prior in $\\tau,f$') # result from this marginalisation
tauq, = plt.plot(X[1,:],ptau/np.sum(ptau)/dtau,'r', label='uniform prior in $\\tau,Q$')  # previous result, marginalising over Q
diff, = plt.plot([],[],' ',label='rel. peak diff. < 1%')
plt.grid(True)
plt.xlabel('characteristic age, $\\tau $, (s)')
plt.ylabel('$\\propto p(\\tau | D)$')
plt.legend(handles=[tauf, tauq, diff])
plt.show()
#fig.savefig(path+'/data/Plots/tauf_tauQ_marginals.png', dpi=300, bbox_inches = "tight")

#%%
tau1 = ptau2
tau2 = ptau
n_ptau2 = Max_Inx(tau1)
n_ptau = Max_Inx(tau2)
diff = X[0][n_ptau2] - X[0][n_ptau]
print(tau1)
print(n_ptau2)
#print(X[0][n_ptau2])
#%% plot rsiduals of normalised marginals for tau 
plt.plot(X[2,:],(ptau2/np.sum(ptau2)/dtau)-(ptau/np.sum(ptau)/dtau))
plt.show()


#%% try the B-f plane
# box centre
B_cent = a*(-(fdot_cent)/(f_cent)**3)**0.5
print(B_cent)
f_cent = f_funct(0)
#B search box size
B_width = 0.1*B_cent
#f search box size
f_width = err[0]*2
boxres = 200 # number of pixels in each axis
B_vals = B_cent + np.linspace(-B_width , B_width, boxres)
f_vals = f_cent + np.linspace(-f_width , f_width, boxres)
X,Y = np.meshgrid(B_vals, f_vals)
Z = logL5(X, Y)
#prob = np.exp( (Z-Z.max()))
prob = np.exp( (Z-Z.max())) # soften the prob by a large factor to let it show better
fig, ax = plt.subplots()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
p = ax.pcolor(X, Y, prob, cmap='jet', vmin=prob.min(), vmax=prob.max())
cb = fig.colorbar(p)
p = ax.pcolor(X, Y, prob, cmap='jet', vmin=prob.min(), vmax=prob.max())
cm = ax.contour(X, Y, prob, (0.1, 0.2, 0.4, 0.66, 0.9),alpha=1,linewidths=2)
ax.clabel(cm, inline=True)
plt.xlabel('magnetic field, $B$, (T)')
plt.ylabel('frequency (Hz)')
plt.show()
#fig.savefig(path+'/data/Plots/B_f.png', dpi=300, bbox_inches = "tight")
# unnormalised marginal for B:
pB = np.sum(prob,1)
fig, ax = plt.subplots()
plt.grid(True)
plt.plot(X[0,:],pB,'k')
plt.xlabel('magnetic field, $B$, (T)')
plt.ylabel('$\\propto p(\\ B | D)$')
plt.show()
fig.savefig(path+'/data/Plots/B_f_marginal.png', dpi=300, bbox_inches = "tight")


#%%Try the Q-B plane, where Q = B**2 * tau =  a/2/f**2
# box centre
Q_cent = a/2/f_cent**2
#B_cent = a*(-(fdot_cent)/f_cent**3)**0.5
#print(B_cent)
#Q search box size
Q_width = 3e-6*Q_cent
#B search box size
B_width = 0.1*B_cent
boxres = 200 # number of pixels in each axis
Q_vals = Q_cent + np.linspace(-Q_width , Q_width, boxres)
B_vals = B_cent + np.linspace(-B_width , B_width, boxres)
X,Y = np.meshgrid(B_vals, Q_vals)
Z = logL6(X, Y)
prob = np.exp( (Z-Z.max()))
fig, ax = plt.subplots()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
p = ax.pcolor(X, Y, prob, cmap='jet', vmin=prob.min(), vmax=prob.max())
cb = fig.colorbar(p)
p = ax.pcolor(X, Y, prob, cmap='jet', vmin=prob.min(), vmax=prob.max())
cm = ax.contour(X, Y, prob, (0.1, 0.5, 0.8),alpha=1,linewidths=2)
ax.clabel(cm)
plt.xlabel('magnetic field, $B$, (T)')
plt.ylabel('$Q$')
plt.show()
fig.savefig(path+'/data/Plots/Q_B.png', dpi=300, bbox_inches = "tight")
# unnormalised marginal for B:
pB1 = np.sum(prob,1)
fig, ax = plt.subplots()
plt.plot(X[0,:],pB1,'k')
plt.grid(True)
plt.xlabel('magnetic field, $B$, (T)')
plt.ylabel('$\\propto p(\\ B | D)$')
plt.show()

#%% plot posteriors of B
dB = X[1,1]-X[1,0]
fig, ax = plt.subplots()
plt.grid(True)
AA, = plt.plot(X[0,:],pB/np.sum(pB)/dB,'b',label='uniform prior in $B,f$')
A, = plt.plot(X[0,:],pB1/np.sum(pB1)/dB,'r',label='uniform prior in $B,Q$')
AAA, = plt.plot([],[], ' ', label = 'rel. peak diff. < 1%')
plt.xlabel('magnetic field, $B$, (T)')
plt.ylabel('$\\propto p(\\ B | D)$')
plt.legend(handles=[A, AA,AAA])
plt.show()
fig.savefig(path+'/data/Plots/Bf_BQ_marginals.png', dpi=300, bbox_inches = "tight")


#%% plot posteriors of B

fig, ax = plt.subplots()
pBf= plt.plot(X[1,:],pB/np.sum(pB)/dB) # result from this marginalisation
pB1Q= plt.plot(X[1,:],pB1/np.sum(pB1)/dB)  # previous result, marginalising over Q
plt.xlabel('characteristic age, $B$, (s)')
plt.ylabel('$p(B | D)$')
plt.show()


#%% try the B-fdot plane
# box centre
B_cent = a*(-fdot_cent/f_cent**3)**0.5
fdot_cent = (f_funct(500)-f_funct(-500))/(1000)
#B search box size
B_width = 0.1*B_cent
#fdot search box size
fdot_width = err[0]*2/t[-1]
boxres = 200 # number of pixels in each axis
B_vals = B_cent + np.linspace(-B_width , B_width, boxres)
fdot_vals = fdot_cent + np.linspace(-fdot_width, fdot_width, boxres)
X,Y = np.meshgrid(B_vals, fdot_vals)
Z = logL7(X, Y)
#prob = np.exp( (Z-Z.max()))
prob = np.exp( (Z-Z.max())/1e9) # soften the prob by a large factor to let it show better
fig, ax = plt.subplots()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
p = ax.pcolor(X, Y, prob, cmap='jet', vmin=prob.min(), vmax=prob.max())
cb = fig.colorbar(p)
p = ax.pcolor(X, Y, prob, cmap='jet', vmin=prob.min(), vmax=prob.max())
cm = ax.contour(X, Y, prob, (0.1, 0.2, 0.4, 0.66, 0.9),alpha=1,linewidths=2)
ax.clabel(cm, inline=True)
plt.xlabel('magnetic field, $B$ (T)')
plt.ylabel('$\dot{f}$ (Hz/s)')
plt.show()
fig.savefig(path+'/data/Plots/B_fdot.png', dpi=300, bbox_inches = "tight")

#%% Try fdot-tau plane
# box centre
tau_cent = -f_cent/2/fdot_cent
fdot_cent = (f_funct(500)-f_funct(-500))/(1000)
#frequency search box size
fdot_width = err[0]*2/t[-1]
#tau search box size
tau_width = 0.1*tau_cent
boxres = 200 # number of pixels in each axis
fdot_vals = fdot_cent + np.linspace(-fdot_width, fdot_width, boxres)
tau_vals = tau_cent + np.linspace(-tau_width, tau_width, boxres)
X,Y = np.meshgrid(tau_vals, fdot_vals)
Z = logL8(X, Y)
prob = np.exp( (Z-Z.max())/1e10)
fig, ax = plt.subplots()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
p = ax.pcolor(X, Y, prob, cmap='jet', vmin=prob.min(), vmax=prob.max())
cb = fig.colorbar(p)
p = ax.pcolor(X, Y, prob, cmap='jet', vmin=prob.min(), vmax=prob.max())
cm = ax.contour(X, Y, prob, (0.1, 0.2, 0.4, 0.66, 0.9),alpha=1,linewidths=2)
ax.clabel(cm, inline=True)
plt.xlabel('characteristic age, $\\tau $ (s)')
plt.ylabel('$\dot{f}$ (Hz/s)')
fig.savefig(path+'/data/Plots/tau_fdot.png', dpi=300, bbox_inches = "tight")
plt.show()


#%% Function for FWHM
def FullWHM(array,x):
    peak = Max_Inx(array) 
    i=0
    l = len(array)
    while i < l :
        if array[i] == array[peak]*0.5 :     
            FWHM_inx = i  
            break 
          
        i += 1    
         
    n1 = peak - FWHM_inx
    n2 = peak + FWHM_inx
    FWHM = x[n2]-x[n1]
    return FWHM

#%%    
ABC = np.asarray([0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0])
XXX = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
print(FullWHM(ABC,XXX))