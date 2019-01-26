#%% import libraries
import matplotlib.pyplot as plt
import numpy as np
from itertools import islice
import operator
import math
import datetime as dt
import textwrap
from mpl_toolkits.mplot3d.axes3d import Axes3D
%matplotlib inline

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
        plt.text((n*T)+10000, 200, textstr, fontsize=20,
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
        plt.text(41000, 200, textstr, fontsize=20,
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
#%% Jordell Bank data plot
plt.plot(jb_data[:,1],jb_data[:,0],'bo-')
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
slope_jb, intercept_jb = np.polyfit(jb_data[:,1], jb_data[:,0], 1)
plt.grid(True)
plt.ylabel('Frequency [Hz]')
plt.xlabel('MJD')
plt.title('Frquency Slow down, Jordell Bank Observatory')
plt.show()
print(slope_jb)

#%% One plot
plt.plot(jd_to_mjd(Jdate_array),Freq_array*1.0001,'ro')
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
ax.errorbar(jd_to_mjd(Jdate_array), Freq_array*1.0001, 1/(4096*10.24),fmt='ro')
plt.grid(True)
slope, intercept = np.polyfit(jd_to_mjd(Jdate_array), Freq_array*1.0001, 1)
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

i=0
while i<len(date_eph) :
    if date_eph[i] == [1992, 3, 5, 12, 46]:
        print(i)
    i=i+1

#%%



k=0
pos=np.zeros(7)
while k<=6:
    j=0
    while j<len(date_eph):
        if date_eph[j] == vars()['date_'+str(k+3)]:
            pos[k]=j
        j=j+1
    k=k+1    

print(pos)











#%% likelyhood functions

err = 1/(4096*10.24)
Freq_corr = Freq_array*1.0001
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

#%% plot likelyhood
#frequency search box size
f_width = err*2
#fdot search box size
fdot_width = err*2/t[-1]
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
cm = ax.contour(X, Y, prob, np.arange(0,1,0.2),alpha=1,linewidths=2)
ax.clabel(cm, inline=True)
cb = fig.colorbar(p)
plt.xlabel('fdot (Hz/s)')
plt.ylabel('frequency (Hz)')
plt.show()

#%%
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.get_yaxis().get_major_formatter().set_useOffset(False)
ax.plot_surface(X, Y, prob, alpha=0.4)
cset = ax.contour(X, Y, prob, zdir='z', cmap='jet')
ax.view_init(40, 30)
plt.show()
# cset = ax.contour(X, Y, Z, zdir='x', cmap='jet')
# cset = ax.contour(X, Y, Z, zdir='y', cmap='jet')