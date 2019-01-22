#%% import libraries
import matplotlib.pyplot as plt
import numpy as np
from itertools import islice
import operator
import math
import datetime as dt
#%% path file for data
path="/Users/federicostachurski/Desktop/ASTRO_Lab_PSR-CRAB/data/"
#%%Read Joerdell Bank observations

jd_data=np.loadtxt(path+"JB_data_1992.cpx")
print(jd_data[:,0])


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

#%%



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
    return year, month, day


print(jd_to_date(par_1[0]))

#%% change mjd into jd
def jd_to_mjd(jd):
     return jd - 2400000.5


#%% Frequency plot
plt.plot(jd_to_mjd(Jdate_array),Freq_array,'ro-')
plt.grid(True)
slope, intercept = np.polyfit(jd_to_mjd(Jdate_array), Freq_array, 1)
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
ax.errorbar(jd_to_mjd(Jdate_array), Freq_array, 1/(4096*10.24),fmt='ro')
plt.ylabel('Frequency [Hz]')
plt.xlabel('MJD')
plt.title('Frquency Slow down')
plt.show()

print(slope)
#%% Jordell Bank data plot
plt.plot(jd_data[:,1],jd_data[:,0],'bo-')
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
slope_jb, intercept_jb = np.polyfit(jd_data[:,1], jd_data[:,0], 1)
plt.grid(True)
plt.ylabel('Frequency [Hz]')
plt.xlabel('MJD')
plt.title('Frquency Slow down, Jordell Bank Observatory')
plt.show()
print(slope_jb)

#%% One plot

plt.plot(jd_to_mjd(Jdate_array),Freq_array,'ro-')
plt.grid(True)
slope, intercept = np.polyfit(jd_to_mjd(Jdate_array), Freq_array, 1)
plt.plot(jd_data[:,1],jd_data[:,0],'bo-')
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
plt.ylabel('Frequency [Hz]')
plt.xlabel('MJD')
plt.title('Frquency Slow down')



#%% METHOD 2: look at entire data

tot_data=np.concatenate([data_1,data_2,data_3,data_4,data_5,data_6,data_7,data_8,data_9],axis=0)
print(tot_data)
#tot_signal=signal_plt(tot_data,par_1,'1 2 3 4',0)
tot_FFT=FastFT(tot_data,par_1,0)

