#%% import libraries
import matplotlib.pyplot as plt
import numpy as np
from itertools import islice
#%% path file for data
path="/Users/federicostachurski/Desktop/ASTRO_Lab_PSR-CRAB/data/"

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
        plt.xlim(0, 40000)
        textstr = '\n'.join((
        r"Obs. points = "+str(n),
        r"Time int. = " +str(T)))
        plt.text(41000, 200, textstr, fontsize=20,
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
    f=np.fft.fftfreq(n,T)
    tot_data=np.zeros(n)+complex(0,0)
    i=0
    while i<n:
        tot_data[i]=np.complex(data[i,0],data[i,1])
        i=i+1
    FFT=np.fft.fft(tot_data)/n
    if inx is 0:
        plt.plot(f,abs(abs(FFT.real)),'r')
        plt.ylabel('Signal Real FFT')
        plt.xlabel('Frequency [Hz]')
        plt.title('Date ' +str(Jdate))
        plt.show()
    elif inx is 1:
        plt.plot(f,abs(abs(FFT.imag)),'b')
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

#%%
