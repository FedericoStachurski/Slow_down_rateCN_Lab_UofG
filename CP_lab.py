#%% import libraries
import matplotlib.pyplot as plt
import numpy as np
from itertools import islice
import operator
import math
import datetime as dt
import textwrap
from mpl_toolkits.mplot3d.axes3d import Axes3D
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body
from astropy.time import Time
solar_system_ephemeris.set('URL: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de200.bsp')
%matplotlib inline

#%% Constants
c = 3*(10**5) #speed of light km/s
R = 1.5*(10**8) # km AU
PSR_EcLon = 83.98316667 #PSR Ecliptic long
PSR_EcLat = -1.29544444 #PSR Ecliptic lat
DM = 56.77118 #Dispersion measure
err_DM = 2.400e-04
DM_const = (4.15*(10**3)) #constant for t_dispersion
Angles = SkyCoord('05 34 31.973 +22 00 52.06',unit=(u.hourangle, u.deg), frame='icrs')  #Right Ascenscion J (SSB) Crab
RAJ = Angles.ra.value # err_RAJ = 5.000e-03  #error RA
DECJ = Angles.dec.value  # err_DECJ = 6.000e-02 #error DEC
Geo_coord=[52.16361111, 0.3722222] #One Mile telescope coordinates
loc=astropy.coordinates.EarthLocation.from_geodetic(0.3722222, 52.16361111, 22)

#%% path file for data
path="/Users/federicostachurski/Desktop/ASTRO_Lab_PSR-CRAB/data/"

#%%Read Joerdell Bank observations
jb_data=np.loadtxt(path+"JB_data_1992.cpx")

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
            verticalalignment='top',transform=ax.transAxes)
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
            verticalalignment='top',transform=ax.transAxes)
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

# print(jd_to_date(par_1[0]))
# print(jd_to_date(par_2[0]))
# print(jd_to_date(par_3[0]))
# print(jd_to_date(par_4[0]))
# print(jd_to_date(par_5[0]))
# print(jd_to_date(par_6[0]))
# print(jd_to_date(par_7[0]))
# print(jd_to_date(par_8[0]))
# print(jd_to_date(par_9[0]))
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


#%% change mjd into jd
def jd_to_mjd(jd):
     return jd - 2400000.5


#%% Frequency plot
plt.plot(jd_to_mjd(Jdate_array),Freq_array*1.0001,'ro')
plt.grid(True)
slope, intercept = np.polyfit(jd_to_mjd(Jdate_array), Freq_array*1.0001, 1)
plt.plot(jd_to_mjd(Jdate_array),(slope*jd_to_mjd(Jdate_array))+intercept, 'g')
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
ax.errorbar(jd_to_mjd(Jdate_array), Freq_array*1.0001, 1/(4096*10.24),fmt='ro')
plt.ylabel('Frequency [Hz]')
plt.xlabel('MJD')
plt.title('Frquency Slow down')
plt.show()
# print(jd_to_date(Jdate_array))

print(slope)


#%% One plot
plt.plot(jd_to_mjd(Jdate_array),Freq_array,'ro')
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
ax.errorbar(jd_to_mjd(Jdate_array), Freq_array, 1/(4096*10.24),fmt='ro')
plt.grid(True)
slope, intercept = np.polyfit(jd_to_mjd(Jdate_array), Freq_array, 1)
plt.plot(jd_to_mjd(Jdate_array),(slope*jd_to_mjd(Jdate_array))+intercept, 'g')
plt.plot(jb_data[:,1],jb_data[:,0],'bo-')
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
textstr = '\n'.join((
r"Slope JB = "+str(float(slope_jb)),
r"Intercept JB = "+str(float(intercept_jb)),
r"Slope Data = " +str(float(slope)),
r"Intercept Data = " +str(float(intercept))))
plt.text(0.8,0.4, textstr,fontsize=11,horizontalalignment='left',transform=ax.transAxes,bbox=dict(facecolor='white', alpha=1))
plt.ylabel('Frequency [Hz]')
plt.xlabel('MJD')
plt.title('Frquency Slow down')

#%% read vector ephemeris (VECTOR) ORBIT
#read Earth-Sun Ephemeris from 1992/3/4 - 1992/3/17
ephe_v_vector=np.genfromtxt(path+"horizons_results_vector.txt", delimiter=',')

v_x_ephe_vector = ephe_v_vector[:,5]
v_y_ephe_vector = ephe_v_vector[:,6]
v_z_ephe_vector = ephe_v_vector[:,7]
jd_vector = ephe_v_vector[:,0]


#%% read vector ephemeris (VECTOR) ROTATIONAL
time=np.array(Jdate_array)
t=astropy.time.Time(t, format='jd')
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

#%% convert into date (VECTOR) orbit
date_eph_vector=list(map(jd_to_date,jd_vector))
date_eph_vector=list(map(list, date_eph_vector))

#%% print(date_eph_vector)
i=0
while i<len(date_eph_vector):
    if date_eph_vector[i][5] == 59 :
        date_eph_vector[i][4] = date_eph_vector[i][4] + 1
    del date_eph_vector[i][-1]
    i=i+1
print(date_eph_vector)

#%% check position (VECTOR)
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

#%% vectors at dates 
v_3 = [v_x[pos_vector[0]], v_y[pos_vector[0]],v_z[pos_vector[0]]]
v_4 = [v_x[pos_vector[1]], v_y[pos_vector[1]],v_z[pos_vector[1]]]
v_5 = [v_x[pos_vector[2]], v_y[pos_vector[2]],v_z[pos_vector[2]]]
v_6 = [v_x[pos_vector[3]], v_y[pos_vector[3]],v_z[pos_vector[3]]]
v_7 = [v_x[pos_vector[4]], v_y[pos_vector[4]],v_z[pos_vector[4]]]
v_8 = [v_x[pos_vector[5]], v_y[pos_vector[5]],v_z[pos_vector[5]]]
v_9 = [v_x[pos_vector[6]], v_y[pos_vector[6]],v_z[pos_vector[6]]]


v_x





#%% Total vector (v_orb + v_rot)
tot_v_x = v_x_ephe_vector + vel_rot_x
tot_v_y = v_y_ephe_vector + vel_rot_y
tot_v_z = v_z_ephe_vector + vel_rot_z
print(v_x_ephe_vector,vel_rot_x,tot_v_x)


#%% unit vector position Crab Nebula 


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




#%% Doppler shift f=f_obs*(1/1-[vcos(wt-lambda)/c])
#maybe add 0.46 in velocity due to earth's rotation (LOS)
def doppler(freq,velocity,angle):
    factor_1 = 1 - ((velocity)/c)*np.sin((angle-83.9831667)*0.017453292519)
    corr_freq = freq * factor
    return corr_freq


corr_freq_3=doppler(Freq_array[0],np.asarray(Vobs_ephe_3),np.asarray(EcLON_ephe_3))
corr_freq_4=doppler(Freq_array[1],np.asarray(Vobs_ephe_4),np.asarray(EcLON_ephe_4))
corr_freq_5=doppler(Freq_array[2],np.asarray(Vobs_ephe_5),np.asarray(EcLON_ephe_5))
corr_freq_6=doppler(Freq_array[3],np.asarray(Vobs_ephe_6),np.asarray(EcLON_ephe_6))
corr_freq_7=doppler(Freq_array[4],np.asarray(Vobs_ephe_7),np.asarray(EcLON_ephe_7))
corr_freq_8=doppler(Freq_array[5],np.asarray(Vobs_ephe_8),np.asarray(EcLON_ephe_8))
corr_freq_9=doppler(Freq_array[6],np.asarray(Vobs_ephe_9),np.asarray(EcLON_ephe_9))

# #%% dot profuct
# angles = [np.cos((np.asarray(EcLON_ephe_3)-83.9831667)*0.017453292519),  np.sin((-1.29544444)*0.017453292519)]
# factor_2 = 1 - np.dot(np.asarray(Vobs_ephe_3) , angles)
# FRQ_DOT_1 = Freq_array[0] * factor_2
# print(corr_freq_3, FRQ_DOT_1)

#%%
i=0
corr_freq_arr=np.zeros(7)
while i<7:
    corr_freq_arr[i]=vars()['corr_freq_'+str(i+3)]
    i=i+1

print(corr_freq_arr)

#%% plot correct (doppler) frequency
plt.plot(jd_to_mjd(Jdate_array),corr_freq_arr,'ro-')
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
ax.errorbar(jd_to_mjd(Jdate_array), corr_freq_arr, 1/(4096*10.24),fmt='ro')
# plt.plot(jd_to_mjd(Jdate_array),Freq_array,'go-')
# ax = plt.gca()
# ax.get_yaxis().get_major_formatter().set_useOffset(False)
# ax.errorbar(jd_to_mjd(Jdate_array), Freq_array, 1/(4096*10.24),fmt='go')
plt.grid(True)
slope, intercept = np.polyfit(jd_to_mjd(Jdate_array), corr_freq_arr, 1)
plt.plot(jd_to_mjd(Jdate_array),(slope*jd_to_mjd(Jdate_array))+intercept, 'g')
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

#%% Frequency plot (Corrected)
plt.plot(jd_to_mjd(Jdate_array),corr_freq_arr,'ro')
plt.grid(True)
slope, intercept = np.polyfit(jd_to_mjd(Jdate_array), corr_freq_arr, 1)
plt.plot(jd_to_mjd(Jdate_array),(slope*jd_to_mjd(Jdate_array))+intercept, 'g')
textstr = '\n'.join((
r"Slope Data (corr) = " +str(np.float32(slope)),
r"Intercept Data (corr) = " +str(np.float32(intercept)),
r"Reduced CHI Squared(corr) = " +str(np.float32(CHI_SQ))))
plt.text(0.6,0.7, textstr,fontsize=15,horizontalalignment='left',transform=ax.transAxes,bbox=dict(facecolor='white', alpha=1))
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
ax.errorbar(jd_to_mjd(Jdate_array), corr_freq_arr, 1/(4096*10.24),fmt='ro')
plt.ylabel('Frequency [Hz]')
plt.xlabel('MJD')
plt.title('Frquency Slow down')
plt.show()





#%% likelyhood functions

err = 1/(4096*10.24)
Freq_corr = corr_freq_arr
epoch = 3 # the index of the epoch of the solution
t = (jd_to_mjd(Jdate_array) - jd_to_mjd(Jdate_array[epoch]))*24*3600
best_f= np.polyfit(t-t[epoch],Freq_corr,1)
f_funct = np.poly1d(best_f)
# print(best_f)
fdot=best_f[0]
f_mu=best_f[1]

def logL1(fdot,f): # the log likelihood as a function of fdot and f
    sum = 0
    for i in range(0,7):
        sum += (Freq_corr[i] - (fdot*t[i] + f))**2/err**2
    return -sum/2
def logL2(tau,B): # the log likelihood as a function of B and tau
# using f = a/B*(2*tau)**(-1/2), fdot = -a/B*(2*tau)**(-3/2) where a = 3.2e-15 and B is in tesla
    sum = 0
    for i in range(0,7):
        sum += (f_obs[i] - (-a/B*(2*tau)**(-3/2)*t[i] + a/B*(2*tau)**(-1/2)))**2/err**2
    return -sum/2


#%% plot likelyhood (f fdot)
#frequency search box size
f_width = err*2
#fdot search box size
fdot_width = err*2/t[-1]
# box centre
f_cent = f_funct(0)
fdot_cent = (f_funct(500)-f_funct(-500))/(1000)
boxres=100 #box resolution
f_vals = f_cent + np.linspace(-f_width , f_width, boxres)
fdot_vals = fdot_cent + np.linspace(-fdot_width, fdot_width, boxres)
X,Y = np.meshgrid(fdot_vals, f_vals)
Z = logL1(X, Y)
prob = np.exp(Z-Z.max())
fig, ax = plt.subplots()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
p = ax.pcolor(X, Y, prob, cmap='jet', vmin=prob.min(), vmax=prob.max())
cm = ax.contour(X, Y, prob, np.arange(0,1,0.2),alpha=1,linewidths=2)
ax.clabel(cm, inline=True)
cb = fig.colorbar(p)
plt.xlabel('fdot (Hz/s)')
plt.ylabel('frequency (Hz)')
plt.show()

# plot Marginals
pfreq = np.sum(prob,1)

plt.plot(Y[:,0],pfreq)
plt.xlabel('frequency (Hz)')
plt.ylabel('$\propto p(f|D)$')
ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_useOffset(False)
plt.show()
pfdot = np.sum(prob,0)
plt.plot(X[0,:],pfdot)
plt.xlabel('fdot (Hz/s)')
plt.ylabel('$\propto p(\dot{f}|D)$')
plt.show()


#%% 3D plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.get_yaxis().get_major_formatter().set_useOffset(False)
ax.plot_surface(X, Y, prob, alpha=0.3)
cset = ax.contour(X, Y, prob, zdir='z', cmap='jet')
ax.view_init(30, 30)
plt.show()

