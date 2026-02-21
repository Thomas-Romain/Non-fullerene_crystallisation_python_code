#%% Import
import numpy as np
import os
import sif_parser      #This is the only thing that might not be necessary, it's useful for viewing sif files that our spectrmeter gives out
                           #That being said, I manage to average the data when it's taken at differing cycle numbers, which wouldn't be possible otherwise
import matplotlib.pyplot as plt
from scipy.stats import norm            #This is being imported after the laser models were introduced. I should go back and remove my Norm function at some point
from scipy.integrate import solve_ivp
from lmfit import Model as MDL  #For fitting TCSPC data?
import sys            #Some models take a very long time to calculate, this is input so I know they're still running even if nothing is happening 
import time

start_time = time.time()

#General definitions
k,OD, ft,R0 =1.3, 2, 150,1.5e-9
h = 6.626e-34
c=2.998e8
wavelength = 800e-9
alpha = 4*np.pi*k/wavelength
tamma = np.logspace(-15,-25,num=500)

#Defining Mike's simple optical model & all associated variables
slice_number = 8
num = np.arange(slice_number) #Proves an arrangement of 1 through to *slice_number* to be used in model2
thickness_xtal = 200e-9
dx = thickness_xtal / slice_number
pulse_time = np.arange(0, 1e-12, 1e-15)
postpulse_time = 2 * np.logspace(-12, -5, 200)
time_PL = np.concatenate((pulse_time, postpulse_time))

#File inputs, input here so you can change what file is being called easily 
FilmOne = r"C:\Users\bn23289\OneDrive - University of Bristol\Desktop\PLQY data\Film samples 1"                          #Filepath for Film Samples 1
FilmTwo = r"C:\Users\bn23289\OneDrive - University of Bristol\Desktop\PLQY data\Film samples 2"                          #Filepath for Film Samples 2
CrystalOne = r"C:\Users\bn23289\OneDrive - University of Bristol\Desktop\PLQY data\Crystal samples 1"                    #Filepath for Crystal Samples 1
CrystalTwo = r"C:\Users\bn23289\OneDrive - University of Bristol\Desktop\PLQY data\Crystal samples 2"                    #Filepath for Crystal Samples 2
TCSPCData = r"C:\Users\bn23289\OneDrive - University of Bristol\Desktop\PLQY data\FLIM data\FLIM data for python.csv"    #Filepath for FLIM data with ultrafast component cut off 
CrystalDecay = r"C:\Users\bn23289\OneDrive - University of Bristol\Desktop\PLQY data\Crystal decay trace"                #Filepath for Crystal decay trace
FilmDecay = r"C:\Users\bn23289\OneDrive - University of Bristol\Desktop\PLQY data\Film decay trace"                      #Filepath for Film decay trace
TCSPCIntensityFile = r"C:\Users\bn23289\OneDrive - University of Bristol\Documents\Data\TCSPC\TEst4Pyth.csv"             #Filepath for high and low fluence TCSPC data

#Rate constants for the triplet-model
krad = 2.3e8 #was 2.3e8 prior to edge crystal adjustment
kisc = 1e8
knr = 1 / (1500e-12) #1500e-12
kradnr = krad + knr+kisc
kb = 1.5e-8
kqr = 1/(1e-1)
ktta = .8e-11
kts = .25*kb
ktr = 1e4
tau = 1e9*(1/kradnr)
L_D = 1e9*np.sqrt(6*kb*1e-6*(1/kradnr)/(np.pi*4*1.5e-9))

#FC model  Original values = 6.3e8, 1/2017e-12,5e12,5.8e12,krad1+knr1, 6.5e-8, 2e-8, kenc1, 7e-11, ktta1, ktta
krad1 = 6.3e8
knr1 = 1 / (2017e-12)
kr1 = 5e12
kcs1 = 5.8e12
kradnr1 = krad1 + knr1
kb1 = 6.5e-8
kenc1 = 2e-8
ksrh1 = kenc1
ktta1 = 7e-11
kTC1 = ktta1
    
#Model5 rate constants :(    original values are 2.3e8, 1e8, 1/(1217e-12), krad2+knr2+kisc2, 1.5e-8,1/(1e-1), .8e-11, .25*kb2, 1e4, 0.1*kb2
krad2 = 2.3e8
kisc2 = 1e8
knr2 = 1 / (1500e-12)
kradnr2 = krad2 + knr2+kisc2
kb2 = 1.5e-8
kqr2 = 1/(1e-1)
ktta2 = .8e-11
kts2 = .25*kb2
ktr2 = 1e4
ksta2 = 0.1*kb2 ##this is the singlet triplet annihilation constant
    
#TestForTamir adjustment
knr3 = 1/(633e-12)
kradnr3 = krad2 + knr3+kisc2  
knr12 = 1/(1500e-12)
kradnr12 = krad1 + knr12
tau = 1e9*(1/kradnr2)
    
##Defining these things for changing plots. Keep these as 0, edit the ones below the loading-the-data section to keep things sensible
ErrorPlot, VaryISC, VaryTTA, VarySTA, VaryKTS, ThreeinOne, TCSPCPlot, BigPlot, TestForTamir, oplot, ModelTrue, Ms, TCSPCIntensity, Kinetics, Tpopplot= 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    
#A few minor things, defining colours and titles for saving the graphs more efficiently (saving values in the 'bigplot' section to a model name)
colourss = [['#fdcc8a','#fc8d59','#e34a33','#b30000','#fef0d9'], #Red/orange    0     A series of colour schemes to plot the graphs with. 
               ['#b3cde3','#8c96c6','#8856a7','#810f7c'], #Purple        1      I couldn't get Mike's code to work, so I manually input hash codes
               ['#c2e699','#78c679','#31a354','#006837'], #Green/yellow  2      This is not ideal, but lets me keep the colours consistant 
               ['#cccccc','#969696','#636363','#252525'], #Noir          3
               ['#b30000','#810f7c','#e34a33','#8856a7']] #New colour purple/red 4
colours = colourss[4]      #Selecting the ones I want for our plotting 
array = ['Thin film','Weighted averages','Triplet-included','Free-charges included', 'singlet-triplet annihilation']
Newmodel= 1 #1 = old model, 2 = new model, 3 = new model + triplets, 4 = new model + triplets + free charges 5 = something fancy that we're working on
titles = ['Red-Orange', 'Purple','Green-Yellow','Noir', 'Purple-Red','singlet-triplet annihilation']
titless = ' Some measurements model '
titl = titles[Newmodel]
ocolour = ['#fff7bc','#fe9929','#ec7014','#cc4c02']  #This is for the populations / kinetics section, gives a few orange and purple colours to plot with
pcolour = ['#efedf5','#807dba','#6a51a3','#54278f']


#%% Definitions and all that jazz 
#Solving for diffusion lengths
def Ld(q):
    Diffusion_Length = (np.sqrt((3*q)/(2*np.pi*R0)))*1e6  #Very quick way to call and calculate the diffusion length calculation because I got
    return Diffusion_Length                               #Tired of writing it out every time

# ODE system
def dP_dt(t, P, i0):        #Change in excitation over time, as described in the SI
    S, T, Ic = P            #This utilises all the decay constants of different pathways, and is called in the triplet model 
    sigma = (150e-15) / (2 * np.sqrt(2 * np.log(2)))
    excitation = i0 * norm.pdf(t, loc=.5e-12, scale=sigma)
    return [
        excitation - (kradnr) * S +  .5*ktta * T**2 - kb * S**2,
        kisc*S+kts*S**2 - ktta*T**2-ktr*T,
        krad * (S)
    ]

def dP_dt2(t, P, i0):      #Similar to the previous dp/dt, it just goes over the change in excitation as the system evolves
    S_hot, S, Ce, Ch, T,Tr, Ic = P     #Uses more rate constants, as there are more pathways involved with free charges
    sigma = (150e-15) / (2 * np.sqrt(2 * np.log(2)))
    excitation = i0 * norm.pdf(t, loc=.5e-12, scale=sigma)
    return [
        excitation - (kr1 + kcs1) * S_hot + 0.5 * ktta1 * T**2 - kb1 * S_hot * (S_hot + S),
        kr1 * S_hot + 0.25 * (kenc1 * Ce * Ch) - (kradnr1 + kb1 * (S + S_hot)) * S,
        kcs1 * S_hot - kenc1 * Ce * Ch - ksrh1 * Ce * Tr,
        kcs1 * S_hot - kenc1 * Ce * Ch - ksrh1 * Ch * Tr,
        .75 * (kenc1 * Ce * Ch) - ktta1 * T**2 - kTC1* T*Ce,
        -ksrh1 * Ch * Tr,
        krad1 * (S + S_hot)
    ]

def dP_dt3(t, P, i0):       #Similar to the previous and previous-previous dp/dt, it just includes triplet-singlet annihilation as well
    S, T, Ic = P
    sigma = (150e-15) / (2 * np.sqrt(2 * np.log(2)))
    excitation = i0 * norm.pdf(t, loc=.5e-12, scale=sigma)
    return [
        excitation - (kradnr2) * S +  .5*ktta2 * T**2 - kb2 * S**2 - ksta2*S*T,
        kisc2*S+kts2*S**2 - ktta2*T**2-ktr2*T,#-ksta*S*T,#kisc*S - ktta*T**2-ktr*T
        krad2 * (S)
    ]

def dP_dt4(t, P, i0):       #Adding this in so I can plot 2 values of knr at once (hopefully)
    S, T, Ic = P
    sigma = (150e-15) / (2 * np.sqrt(2 * np.log(2)))
    excitation = i0 * norm.pdf(t, loc=.5e-12, scale=sigma)
    return [
        excitation - (kradnr3) * S +  .5*ktta2 * T**2 - kb2 * S**2 - ksta2*S*T,
        kisc2*S+kts2*S**2 - ktta2*T**2-ktr2*T,#-ksta*S*T,#kisc*S - ktta*T**2-ktr*T
        krad2 * (S)
    ]

def dP_dt5(t, P, i0):      #Adding this so I can plot 2 values of knr at once (but using model 2 instead of model 5)
    S_hot, S, Ce, Ch, T,Tr, Ic = P     #Uses more rate constants, as there are more pathways involved with free charges
    sigma = (150e-15) / (2 * np.sqrt(2 * np.log(2)))
    excitation = i0 * norm.pdf(t, loc=.5e-12, scale=sigma)
    return [
        excitation - (kr1 + kcs1) * S_hot + 0.5 * ktta1 * T**2 - kb1 * S_hot * (S_hot + S),
        kr1 * S_hot + 0.25 * (kenc1 * Ce * Ch) - (kradnr12 + kb1 * (S + S_hot)) * S,
        kcs1 * S_hot - kenc1 * Ce * Ch - ksrh1 * Ce * Tr,
        kcs1 * S_hot - kenc1 * Ce * Ch - ksrh1 * Ch * Tr,
        .75 * (kenc1 * Ce * Ch) - ktta1 * T**2 - kTC1* T*Ce,
        -ksrh1 * Ch * Tr,
        krad1 * (S + S_hot)
    ]

def dP_dt6(t, P, i0):   #Final dp/dt, used in the population plotting section
    S, T, Ic = P
    #sigma = (600e-12) / (2 * np.sqrt(2 * np.log(2)))
    #excitation = i0 * norm.pdf(t, loc=.5e-9, scale=sigma)
    sigma = (150e-15) / (2 * np.sqrt(2 * np.log(2)))
    excitation = i0 * norm.pdf(t, loc=.5e-12, scale=sigma)
    return [
        excitation - (kradnr) * S +  .5*ktta * T**2 - kb * S**2,
        kisc*S+kts*S**2 - ktta*T**2-ktr*T,#kisc*S - ktta*T**2-ktr*T
        krad * (S)
    ]

#Just something to plot the data, for all models / datapoints (not used as much as I initially thought, but removing is too much work for this late into the project)
def Plot(data, model, fluence, title, Diffusion_length, colour, Simulated):    
    ax.scatter(fluence, data, color = colour, label = title) #+ ' data')
    ax.plot(Simulated, model, color = colour, label = 'model of '+ str(round(int(Diffusion_length),-1)) + ' nm $L_{D}$')
    
#A shortcut to normalise something 
def Norm(x):
    x1 = (x - x.min(0))/np.ptp(x, 0) #changed from x.ptp(0) as I was using numpy 1.26.4 or something like that, and x.ptp(0) was changed to be np.ptp(x,0) instead. Have made it so it will run on modern numpy
    # x1 = ((x-x.min(0))/(x[0]))
    return x1

#Exponential decay fitting for TCSPC plot, need to find a way to use this on the tri-exponential non-irf-deconvoluted files?
def multi_exp(t, A1, tau1, A2, tau2, C):
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)+ C

#Calculate carrier densities, be careful with units
def Den(x):
    y= x*1e-6*wavelength*(1-10**(-OD))/(h*c*ft*1e-7)
    return y

#Does a minor adjustment of the datapoints where a change in power-meter sensitivity is visible, 99% of the time this is not called
def SenAdj(Unadjusted_Analysis, Point1, Point2, Newlist, peak1, peak2):
    T1 = sum(np.transpose((Unadjusted_Analysis)[Point1:Point2]))
    T2 = T1[0]/T1[1]
    T3 = np.transpose((Unadjusted_Analysis)[0:Point1])
    T4 = np.transpose((T2*Unadjusted_Analysis)[Point1:])
    v = sum((np.concatenate((T3, T4), axis=1))[peak1:peak2])
    Analysis3 = (v/(Newlist))/(v[0]/(Newlist)[0])
    return Analysis3

#Reads off file path, does analysis / adjustments, then does modelling based on 'NewModel' variable, then plots both data and model. Probably overengineered
#This uses os.walk to direct to your file path so you don't need to move the code into the folder to run it, just input the path of the folder
def Read(filepath, pump1, pump2, flat1, flat2, peak1, peak2, spotsize_a, spotsize_b, Sensitivity_adjust, title, D1, D2, NewModel, colour, normv, time_PL):
    Pump, Peakav, Newlist, Flatadj = [],[],[],[]
    File_Path = filepath
    for directory, subdirectories, files in os.walk(File_Path):
        for file in files:
                    u = file.rsplit('.', 1)[0]
                    filename = filepath + "\\" + file     
                    data, info = sif_parser.utils.parse(filename)               #This reads the sif file with the name found in the file
                    data = data[:,1]/(info['AccumulatedCycles'])                #This normalises the data depending on how many cycles it has done
                    wavelengths = sif_parser.utils.extract_calibration(info)    #This reads the wavelengths that the spectrometer spat out
                    Flat = data-(np.average(data[flat1:flat2,]))                #This is calculating the average background level and adjusting for that
                    Pump = np.append(Pump, np.average(Flat[pump1:pump2]))       #This is getting the values for pump scatter from the adjusted values
                    Peakav = np.append(Peakav, (np.average(Flat[peak1:peak2,])))#This is getting the values for the PL peak from the adjusted values
                    Newlist = np.append(Newlist, int(u))                        #This is getting a value of the pump power from the file-name (e.g. 000100 = 100nw)
                    Flatadj = np.append(Flatadj, Flat)                          #This is combining the adjusted values into one variable
                    # plt.plot(wavelengths,Flat)
                    # plt.show()
    Flatadj = np.reshape(Flatadj, (len(files), len(Flat)))                      #Changing array shapes
    Analysis = (Peakav / Newlist)/(Peakav[0]/Newlist[0])                        #Changing array shapes
    if Sensitivity_adjust == True:
        Analysis = SenAdj(Flatadj, D1, D2, Newlist, peak1, peak2)
    Fluence = ((Newlist*1e-3)/1000)/(np.pi*spotsize_a*1e-4*spotsize_b*1e-4)    #Calculates fluence based on input variables like spot size and power
    Flue = Fluence/((h*c/wavelength))                                          #Is overwritten on the next line so doesn't matter. Struggling with units.
    Flue = ((Fluence*1e4* 1e-6)/ (h * c / wavelength))                         #Something? Is this even called (in something that's used, not model 2?)
    Dens = Den(Fluence) #Carrier densities in m-3                              #Calculates densities using the Dens function defined earlier
    fluence_sim_cm_uJ = np.logspace(-2, 3, len(Analysis))                      #Simulated values of fluence used in plotting smoother lines for the models
    fluence_sim = fluence_sim_cm_uJ # Convert µJ/cm² to J/m²                   #Doesn't do anything?
    N_0 = (fluence_sim)/ (h * c / wavelength)                                  #
    DensSim = Den(N_0* 1e-20)

#Based on input value, it will plot this data using one of 4 models: normal OD, theorised better model, model with triplets, and model with free charges
    if NewModel == 1:
        Model(Dens, Norm(Analysis), tamma, title, Fluence, colour, DensSim, fluence_sim_cm_uJ, normv) #Inputting the calculated densities, analysis, tamma values, fluence and name of plot into model    
    if NewModel == 2:
        Model2(Flue, Norm(Analysis),tamma, Fluence, title, colour, N_0, fluence_sim_cm_uJ, normv) #Inputting the calculated densities, analysis, tamma values, fluence and name of plot into model
    if NewModel == 3:
        Model3(Newlist, Norm(Analysis), title, Fluence, colour, spotsize_a, spotsize_b, fluence_sim, fluence_sim_cm_uJ, time_PL)
    if NewModel == 4:
        Model4(Newlist, Norm(Analysis), title, Fluence, colour, spotsize_a, spotsize_b, fluence_sim, fluence_sim_cm_uJ, time_PL)
    if NewModel == 5:
        Model5(Newlist, Norm(Analysis), title, Fluence, colour, spotsize_a, spotsize_b, fluence_sim, fluence_sim_cm_uJ, time_PL)        
    if ThreeinOne == 1:
        Model(Dens, Norm(Analysis), tamma, title, Fluence, colour, DensSim, fluence_sim_cm_uJ, normv)
        Model3(Newlist, Norm(Analysis), title, Fluence, colour, spotsize_a, spotsize_b, fluence_sim, fluence_sim_cm_uJ, time_PL)
        Model4(Newlist, Norm(Analysis), title, Fluence, colour, spotsize_a, spotsize_b, fluence_sim, fluence_sim_cm_uJ, time_PL)
    return Pump, Peakav, Newlist, Flatadj, wavelengths, Norm(Analysis), Flue, Dens, peak1, peak2, Fluence
#          z0      z1      z2       z3         z4              z5        z6     z7    z8     z9     z10

#Models the old thin-film approximation. Worth noting that the model is called and the error is calculated, and then a simulated model is plotted
#This isn't how I originally wanted to do it but the actual modelled values were based on datapoints, so the line wasn't presentable (had visible jags)
def Model(Densities, Analysis, tammavalues, title, Fluenceujcm, colour, Simulated, fluence_sim_cm_uJ, normv):
    t = np.array([[((np.log(1+(x*y)))/(x*y)) for x in Densities] for y in tammavalues])
    Eran = np.array([sum((Analysis-(Norm(Z)))**2) for Z in t])
    minmodel = np.transpose(t[(np.where(Eran == Eran.min())[0])])
    mintam = tamma[(np.where(Eran == Eran.min())[0])]
    Diffusion_length = Ld(mintam)
    t1 = np.array([((np.log(1+(x*mintam)))/(x*mintam)) for x in Simulated*11])
    
    #Plotting for error analysis 
    if ErrorPlot ==1:
        plt.plot(Ld(tammavalues), Eran, label = title + ' fitting error analysis', color = colour, lw=2)  
        print(title)
        x=2 #Adjust this to adjust error values. When X=2, the plusminus is when the calculated errors are double the minimum. X=2 is used for our reported errors
        Plusminus = Ld(tamma[np.where(Eran < Eran.min()*x)][0]) - Ld(tamma[np.where(Eran < Eran.min()*x)][-1])
        print(Plusminus/2) #This is the value of the +/- reported in the paper
        # print(Ld(tamma[np.where(Eran < Eran.min()*x)][0])) #Lower bound, hashed out
        # print(Ld(tamma[np.where(Eran < Eran.min()*x)][-1])) #Upper bound, hashed out
        print(Eran[np.where(Eran < Eran.min()*x)[0][0]])    #Prints the [0] array value from when the Eran value is double the minimum error value
        plt.scatter(Ld(tamma[np.where(Eran < Eran.min()*x)][0]),Eran[np.where(Eran < Eran.min()*x)[0][0]],s=30, color = colour) #Lower error bound
        plt.scatter(Ld(tamma[np.where(Eran < Eran.min()*x)][-1]),Eran[np.where(Eran < Eran.min()*x)[0][-1]],s=30, color = colour) #Upper error bound
        plt.fill_between(Ld(tamma[np.where(Eran == Eran.min()*x)]), Eran[np.where(Eran < Eran.min()*x)[0][0]], Eran[np.where(Eran < Eran.min()*x)[0][-1]], color='black', alpha=0) #This line doesn't do anything
                                                                                                                                                                                   #, or if it does it's not visible
        
    if ThreeinOne ==0 and ErrorPlot ==0 and TestForTamir ==0: #When we're not doing ThreeinOne or Errorplot we plot normally ala:
        Plot(Analysis, Norm(t1)*normv, Fluenceujcm,title, Diffusion_length, colour, fluence_sim_cm_uJ*0.4)        
    if ThreeinOne ==1: #Plotting only a line using the singlet only model
        ax.plot(fluence_sim_cm_uJ*0.38, Norm(t1)*normv, color = '#feb24c', label ='Singlets-only model',zorder=0)
    if TestForTamir ==1: #Plotting 50nm and 70nm plots on crystal to show difference
        ax.scatter(Fluenceujcm, Analysis, color = '#b30000', label = title) #+ ' data')
        ax.plot(fluence_sim_cm_uJ*0.4, Norm(t1)*normv, color = '#810f7c', label = 'model of '+ str(round(int(Diffusion_length),-1)) + ' nm $L_{D}$')
        t1 = np.array([((np.log(1+(x*8e-18)))/(x*8e-18)) for x in Simulated*11]) #8e-18 gives 50nm LD
        Diffusion_length = Ld(8e-18)
        ax.plot(fluence_sim_cm_uJ*0.4, Norm(t1)*normv, color = '#006837', label = 'model of '+ str(round(int(Diffusion_length),-1)) + ' nm $L_{D}$')
    return  t, minmodel, mintam, Diffusion_length

#New weighted average modelling function, not used in paper as results were not as relevant as first imagined
def Optical_Model(N_0,tamma,dx,alpha):
    eta_rho = []
    eta_rho_i = []
    for N_0_j in N_0:   
        for ii in num:
            rho_0_i = N_0_j*np.exp(-alpha*(ii)*dx)*(1-np.exp(-alpha*dx))/dx
            eta_rho_ii = np.log(1+rho_0_i*tamma)*np.exp(-alpha*ii*dx)/(rho_0_i*tamma)
            eta_rho_i = np.append(eta_rho_i,eta_rho_ii)
    eta_rho_index = eta_rho_i.reshape(len(N_0),len(num))
    eta_rho = np.sum(eta_rho_index, axis = 1)
    return eta_rho

#Models the new thick-film approximation, basic biexponential decay pathways. #Fluences are even more complicated, but Mike can plot them with rate constants
def Model2(Fluence, Analysis, tammavalues, Fluenceujcm, title, colour, Simulated, fluence_sim_cm_uJ, normv):  
    t = [Norm(Optical_Model(Fluence, y, dx, alpha)) for y in tammavalues] #t = list of model values gotten from Optical_Model() code
    Eran = np.array([sum((Analysis-q)**2) for q in t])
    minmodel = t[int((np.where(Eran == Eran.min())[0]))]
    mintam = tammavalues[(np.where(Eran == Eran.min())[0])]
    Diffusion_length = (np.sqrt((mintam*6)/(4*np.pi*R0)))*1e9
    t1 = Optical_Model(Simulated*0.01, mintam, dx, alpha)
    Plot(Analysis, Norm(t1)*normv, Fluenceujcm, title, Diffusion_length, colour, fluence_sim_cm_uJ)
    return t, minmodel, mintam, Diffusion_length

#New-new weighted average modelling thing with triplets included (I do not know how this works do not ask me (I have to simulate fluences))
def Model3(Pre_adj_fluence, Analysis, title, Fluenceujcm, colour,S1,S2, Simulated, fluence_sim_cm_uJ, time_PL):
        slice_number = 3
        num = np.arange(slice_number)
        thickness_xtal = 160e-9
        dx = thickness_xtal / slice_number
        N_0 = ((((Simulated*1e-9)/100)/(np.pi*S1*1e-4*S2*1e-4)) / (h * c / wavelength))
        P0 = [0,0,0]
        eta_rho = [] # Compute eta_rho
        for N_0_j in N_0: ##note N-0 is now in carriers per cubic cm! (to fit with the rate constants)
            eta_rho_i = []
            for ii in num:
                rho_0_i = N_0_j * np.exp(-alpha * ii * dx) * (1 - np.exp(-alpha * dx)) / dx
                sol = solve_ivp(dP_dt, [time_PL[0], time_PL[-1]], P0, t_eval=time_PL, args=(rho_0_i,), method='BDF')
                Ps = sol.y.T
                PL = max(Ps[:, 2])/ rho_0_i
                eta_rho_ii = np.exp(-alpha * ii * dx) * PL
                eta_rho_i.append(eta_rho_ii)
            eta_rho.append(np.sum(eta_rho_i))
        ModelFin = np.array(eta_rho)/max(eta_rho)
        Diffusion_length = 1e9*np.sqrt(6*kb*1e-6*(1/kradnr)/(np.pi*4*1.5e-9))
        print(Diffusion_length)
    
        ##Little addition for the 'varying K value' graphs, checks if we're varying graphs (through Vary___) and if we are, it doesn't do the normal plot
        if VaryISC ==0 and VaryTTA == 0 and VaryKTS==0:
            if ThreeinOne ==0:
                Plot(Analysis, Norm(ModelFin), Fluenceujcm, title, Diffusion_length, colour, fluence_sim_cm_uJ*0.4)
            if ThreeinOne ==1:
                ax.plot(fluence_sim_cm_uJ*0.38, Norm(ModelFin), color = '#fd8d3c', label ='Singlet + triplet model',zorder=0)
        if VaryTTA == 1:
            stupid  = ktta/Placeholder
            ax.plot(fluence_sim_cm_uJ*0.38, Norm(ModelFin), color = colour, label =str(float("%.2g" % stupid)),zorder=0)
        if VaryISC == 1:
            stupid  = kisc/Placeholder
            ax.plot(fluence_sim_cm_uJ*0.38, Norm(ModelFin), color = colour, label =str(float("%.2g" % stupid)),zorder=0)
        if VaryKTS == 1:
            stupid  = kts/Placeholder
            ax.plot(fluence_sim_cm_uJ*0.38, Norm(ModelFin), color = colour, label =str(float("%.2g" % stupid)),zorder=0)
        return eta_rho, ModelFin

#New-new weighted average modelling thing with triplets and free charges included (I also do not know how this works do not ask me(similarly I have to simulate fluences))
def Model4(Pre_adj_fluence, Analysis, title, Fluenceujcm, colour,S1,S2, Simulated, fluence_sim_cm_uJ, time_PL):
    if TestForTamir==1:
        slice_number = 3
        num = np.arange(slice_number)
        thickness_xtal = 160e-9
        dx = thickness_xtal / slice_number
        N_0 = ((((Simulated*1e-9)/100)/(np.pi*S1*1e-4*S2*1e-4)) / (h * c / wavelength))
        Q0=0
        # Initial conditions
        P0 = [0, 0, 0, 0, 0, Q0, 0]
        # Compute eta_rho
        eta_rho = []
        for N_0_j in N_0: ##note N-0 is now in carriers per cubic cm! (to fit with the rate constants)
            eta_rho_i = []
            for ii in num:
                rho_0_i = N_0_j * np.exp(-alpha * ii * dx) * (1 - np.exp(-alpha * dx)) / dx
                sol = solve_ivp(dP_dt2, [time_PL[0], time_PL[-1]], P0, t_eval=time_PL, args=(rho_0_i,), method='BDF')
                Ps = sol.y.T
                PL = max(Ps[:, 6])/ rho_0_i
                eta_rho_ii = np.exp(-alpha * ii * dx) * PL
                eta_rho_i.append(eta_rho_ii)
            eta_rho.append(np.sum(eta_rho_i))
        ModelFin = np.array(eta_rho)/max(eta_rho)
        Diffusion_length = 1e9*np.sqrt(6*kb1*1e-6*(1/kradnr1)/(np.pi*4*1.5e-9))
        print(Diffusion_length)
        ax.scatter(Fluenceujcm, Analysis, color = '#b30000', label = title) #+ ' data')
        ax.plot(fluence_sim_cm_uJ*0.4, Norm(ModelFin), color = '#cccccc', label = 'model of '+ str(round(int(Diffusion_length),-1)) + ' nm $L_{D}$')
        
        eta_rho = []
        for N_0_j in N_0: ##note N-0 is now in carriers per cubic cm! (to fit with the rate constants)
            eta_rho_i = []
            for ii in num:
                rho_0_i = N_0_j * np.exp(-alpha * ii * dx) * (1 - np.exp(-alpha * dx)) / dx
                sol = solve_ivp(dP_dt5, [time_PL[0], time_PL[-1]], P0, t_eval=time_PL, args=(rho_0_i,), method='BDF')
                Ps = sol.y.T
                PL = max(Ps[:, 6])/ rho_0_i
                eta_rho_ii = np.exp(-alpha * ii * dx) * PL
                eta_rho_i.append(eta_rho_ii)
            eta_rho.append(np.sum(eta_rho_i))
        ModelFin = np.array(eta_rho)/max(eta_rho)
        Diffusion_length = 1e9*np.sqrt(6*kb1*1e-6*(1/kradnr12)/(np.pi*4*1.5e-9))
        print(Diffusion_length)
        ax.plot(fluence_sim_cm_uJ*0.4, Norm(ModelFin), color = '#006837', label = 'model of '+ str(round(int(Diffusion_length),-1)) + ' nm $L_{D}$')
        
    else:
        slice_number = 3
        num = np.arange(slice_number)
        thickness_xtal = 160e-9
        dx = thickness_xtal / slice_number
        N_0 = ((((Simulated*1e-9)/1000)/(np.pi*S1*1e-4*S2*1e-4)) / (h * c / wavelength))*13
        Q0=0
        P0 = [0, 0, 0, 0, 0, Q0, 0]
        eta_rho = []
        for N_0_j in N_0: ##note N-0 is now in carriers per cubic cm! (to fit with the rate constants)
            eta_rho_i = []
            for ii in num:
                rho_0_i = N_0_j * np.exp(-alpha * ii * dx) * (1 - np.exp(-alpha * dx)) / dx
                sol = solve_ivp(dP_dt2, [time_PL[0], time_PL[-1]], P0, t_eval=time_PL, args=(rho_0_i,), method='BDF')
                Ps = sol.y.T
                PL = max(Ps[:, 6])/ rho_0_i
                eta_rho_ii = np.exp(-alpha * ii * dx) * PL
                eta_rho_i.append(eta_rho_ii)
            eta_rho.append(np.sum(eta_rho_i))
        ModelFin = np.array(eta_rho)/max(eta_rho)
        Diffusion_length = 1e9*np.sqrt(6*kb*1e-6*(1/kradnr)/(np.pi*4*1.5e-9))
        if ThreeinOne ==0:
            Plot(Analysis, Norm(ModelFin), Fluenceujcm, title, Diffusion_length, colour, fluence_sim_cm_uJ*0.38) #Values are plot with a simulated fluence, not what
        if ThreeinOne ==1:                                                                                       #was used in the model calculation
            ax.plot(fluence_sim_cm_uJ*0.38, Norm(ModelFin), color = '#f03b20', label ='Free-charge model',zorder=0)
        return eta_rho, ModelFin                                                                             

#Singlet-triplet annihilation model, useful for showing the effect that taking this into account has (horribly phrased)
def Model5(Pre_adj_fluence, Analysis, title, Fluenceujcm, colour,S1,S2, Simulated, fluence_sim_cm_uJ, time_PL):
    if TestForTamir==1:
        slice_number = 3
        num = np.arange(slice_number)
        thickness_xtal = 160e-9
        dx = thickness_xtal / slice_number
        N_0 = ((((Simulated*1e-9)/100)/(np.pi*S1*1e-4*S2*1e-4)) / (h * c / wavelength))
        P0 = [0,0,0]
        eta_rho = []
        for N_0_j in N_0: 
            eta_rho_i = []
            for ii in num:
                rho_0_i = N_0_j * np.exp(-alpha * ii * dx) * (1 - np.exp(-alpha * dx)) / dx
                sol = solve_ivp(dP_dt3, [time_PL[0], time_PL[-1]], P0, t_eval=time_PL, args=(rho_0_i,), method='BDF')
                Ps = sol.y.T
                PL = max(Ps[:, 2])/ rho_0_i
                eta_rho_ii = np.exp(-alpha * ii * dx) * PL
                eta_rho_i.append(eta_rho_ii)
            eta_rho.append(np.sum(eta_rho_i))
        ModelFin = np.array(eta_rho)/max(eta_rho)
        Diffusion_length = 1e9*np.sqrt(6*kb2*1e-6*(1/kradnr2)/(np.pi*4*1.5e-9))
        print(Diffusion_length)
        ax.scatter(Fluenceujcm, Analysis, color = '#b30000', label = title) #+ ' data')
        ax.plot(fluence_sim_cm_uJ*0.4, Norm(ModelFin), color = '#810f7c', label = 'model of '+ str(round(int(Diffusion_length),-1)) + ' nm $L_{D}$')

        eta_rho = []
        for N_0_j in N_0: 
            eta_rho_i = []
            for ii in num:
                rho_0_i = N_0_j * np.exp(-alpha * ii * dx) * (1 - np.exp(-alpha * dx)) / dx
                sol = solve_ivp(dP_dt4, [time_PL[0], time_PL[-1]], P0, t_eval=time_PL, args=(rho_0_i,), method='BDF')
                Ps = sol.y.T
                PL = max(Ps[:, 2])/ rho_0_i
                eta_rho_ii = np.exp(-alpha * ii * dx) * PL
                eta_rho_i.append(eta_rho_ii)
            eta_rho.append(np.sum(eta_rho_i))
        ModelFin = np.array(eta_rho)/max(eta_rho)
        Diffusion_length = 1e9*np.sqrt(6*kb2*1e-6*(1/kradnr3)/(np.pi*4*1.5e-9))
        print(Diffusion_length)
        
        ax.plot(fluence_sim_cm_uJ*0.38, Norm(ModelFin), color = '#006837', label = 'model of '+ str(round(int(Diffusion_length),-1)) + ' nm $L_{D}$')
        
    else:
        slice_number = 3
        num = np.arange(slice_number)
        thickness_xtal = 160e-9
        dx = thickness_xtal / slice_number                                                                   
        N_0 = ((((Simulated*1e-9)/100)/(np.pi*S1*1e-4*S2*1e-4)) / (h * c / wavelength))
        P0 = [0,0,0]
        eta_rho = []
        for N_0_j in N_0: 
            eta_rho_i = []
            for ii in num:
                rho_0_i = N_0_j * np.exp(-alpha * ii * dx) * (1 - np.exp(-alpha * dx)) / dx
                sol = solve_ivp(dP_dt3, [time_PL[0], time_PL[-1]], P0, t_eval=time_PL, args=(rho_0_i,), method='BDF')
                Ps = sol.y.T
                PL = max(Ps[:, 2])/ rho_0_i
                eta_rho_ii = np.exp(-alpha * ii * dx) * PL
                eta_rho_i.append(eta_rho_ii)
            eta_rho.append(np.sum(eta_rho_i))
        ModelFin = np.array(eta_rho)/max(eta_rho)
        Diffusion_length = 1e9*np.sqrt(6*kb2*1e-6*(1/kradnr2)/(np.pi*4*1.5e-9))
        print(Diffusion_length)
       ##Similar to model 3, checks if we're varying and if we are, doesnt' do the normal plot
        if VarySTA == 0:    
            Plot(Analysis, Norm(ModelFin), Fluenceujcm, title, Diffusion_length, colour, fluence_sim_cm_uJ*0.38)
        else:
            stupid  = ksta2/kb2   ##This is the only way I could figure out how to give the multiplicative value (or the way I found before I gave up)
            ax.plot(fluence_sim_cm_uJ*0.38, Norm(ModelFin), color = colour, label = str(float("%.2g" % stupid)), zorder=0)

#This plots a series of different graphs of all the data (film 1+2, cystal 1+2) on one graph, and with every colour scheme
#The FC and triplet models take a long time to run, so I wrote this to plot all different versions of the graphs with one button press, so I could grab coffee
#While the code is running. Terrible and barely used: possibly remove?
def SavePlot(x, titles, titless, Newmodel, number):
    colours = colourss[x]
    titl = titles[x]
    TTitle = ['Cannot-be-called-no-0-model','Singlet-only','Optical depth','Triplet-included','Free-charge model','singlet-triplet annihilation model']
    
    ## fig = plt.figure()            
    ## ax = fig.add_subplot(111)
    ## fig.figsize=(20,12)
    
    #######Hash and unhash specific files when needed
    # title, colour = 'Film sample 1', colours[1]
    # Read(FilmTwo, 200, 230,1400,1600, 500,800, 150, 150, True, title, 26, 28,#Pump1, pump2, flat1, flat2, peak1, peak2, spotsize_a and spotsize_b, sen_adj and the title
    #           Newmodel, colour, 1.05, time_PL)  
    
    # title, colour =  'Film sample 2', colours[3]
    # Read(FilmOne,0, 1,1200,1600, 200,500, 330,430, False, title, 0,0,#Pump1, pump2, flat1, flat2, peak1, peak2, spotsize_a and spotsize_b, sen_adj and the title
    #             Newmodel, colour, 1.1, time_PL)  
    
    # title, colour =  'Crystal samples 2', colours[2]
    # Read(CrytalTwo,0, 1,1400,1600, 300,600, 130,200, True, title, 13, 15, #Pump1, pump2, flat1, flat2, peak1, peak2, spotsize_a and spotsize_b, sen_adj and the title
    #           Newmodel, colour, 1.15, time_PL)  

    title, colour = 'Crystal samples 1', colours[3]
    Read(CrystalOne, 0, 80,1200,1600, 300,600, 300,300, False, title, 0,0,#Pump1, pump2, flat1, flat2, peak1, peak2, spotsize_a and spotsize_b, sen_adj and the title
              Newmodel, colour, 1, time_PL) 
    
    #Legend adjustment for SI plot
    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [2,3,0,1,4,5]
    # ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],frameon=False, ncol=1, title='Singlet-only model') 
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    # ax.legend(frameon = False, ncol = 1, title=str(array[int((Newmodel)-1)]) + ' model')#loc='lower right'
    ax.legend(frameon=False, ncol=1, title = str(TTitle[Newmodel]))
    
    plt.xscale('log'),plt.xlabel('Fluence (μJ cm\u207B\u00B2)' ),plt.ylabel('ηPL/ηPL\u2080 (A.U)')
    plt.ylim(0, 1), plt.xlim(0.1, 1000)
    # fig.savefig('Paperfigures'+ titl + titless + str(number)+ '.svg',bbox_inches='tight')
    # fig.savefig('v with space'+ titl + titless + str(number)+ '.png',bbox_inches='tight') #Save functionality, hash out when not using
    plt.show()

#This plots the decay traces and provides a rough plot for the eye, using the same nonlinear fitting as the TCSPC code. Probably over-engineered, but it was
#mostly copy-pasting and re-jigging some numbers
def MS(File_Path, title, colour, Tt,Normal,Tau1val):
    Pump, Peakav,Val, Newlist= [], [],[],[]
    for directory, subdirectories, files in os.walk(File_Path):
            for file in files:
                        u = file.rsplit('.', 1)[0]
                        filename = File_Path + "\\" + file     
                        data, info = sif_parser.utils.parse(filename)               #This reads the sif file with the name found in the file
                        data = data[:,1]/(info['AccumulatedCycles'])                #This normalises the data depending on how many cycles it has done
                        Flat = data-(np.average(data[1400:1600,]))                #This is calculating the average background level and adjusting for that
                        Pump = np.append(Pump, np.average(Flat[0:1]))       #This is getting the values for pump scatter from the adjusted values
                        Peakav = np.append(Peakav, (np.average(Flat[1:2,])))#This is getting the values for the PL peak from the adjusted values
                        Newlist = np.append(Newlist, int(u))                        #This is getting a value of the pump power from the file-name (e.g. 000100 = 100nw)
                        Val = np.append(Val, np.sum(Flat[220:400]))                          #This is combining the adjusted values into one variable
    
    params = MDL(multi_exp).make_params(A1=2, tau1=Tau1val, A2=1, tau2=Tt, C=0)
    if Normal == True:
        plt.scatter(Newlist, Norm(Val/Val[0]),label = title,color=colour)
        plt.plot(Newlist, (MDL(multi_exp).fit(Norm(Val/Val[0]), params, t=Newlist)).best_fit, color = colour, label=(title+' decay trace'))
        
    else:
        plt.scatter(Newlist, Val/Val[0],label = title,color=colour)
        plt.plot(Newlist, (MDL(multi_exp).fit(Val/Val[0], params, t=Newlist)).best_fit, color = colour, label=(title+' decay trace'))

    plt.xlabel('Time (seconds)')
    plt.ylabel('Relative PL intensity (A.U)')
    plt.xlim(0,5000), plt.ylim(0.4,1)
    plt.legend()
    if Normal == True:
        Fig.savefig('DecayTraces ' + '.svg',bbox_inches='tight')
    else:
        Fig.savefig('DecayTraces non norm' + '.svg',bbox_inches='tight')

#Defining fig and ax for the defined Plot variable thingy haha look at me I almost know what I'm talking about 
def Figload():
    plt.figure(figsize=(10,6)) 
    Fig = plt.figure()
    ax = Fig.add_subplot(111)
    Fig.figsize=(20,12)
    plt.rcParams.update({'font.size': 12})
    return(ax, Fig)

#%% Actual plotting / analysis
##Un-hashing these will allow the 'read' function to be called and perform all the analysis / plotting
#I understand calling the files every time isn't ideal, and that Jupyter can hold the values in memory, but I don't like Jupyter
#You have to input the array values you want to use as your start and end positions for each set of analyses, if that makes sense
#Such as putting in the start and end values for the location of the peak, on that specific set 
# BigPlot = 1
# TestForTamir = 1 #Very quick thing for the SI showing a 50 and 70 nm fit for the crystal data, should remove at some point? Makes things complicated
if BigPlot ==1: 
    ax, Fig = Figload()
    Newmodel = 1
    title, colour = 'Film samples 1', colours[1]
    z = Read(FilmOne, 200, 230,1400,1600, 500,800, 150, 150, True, title, 26, 28,#Pump1, pump2, flat1, flat2, peak1, peak2, spotsize_a and spotsize_b, sen_adj and the title
              Newmodel, colour, 1.05, time_PL) 
    
    title, colour =  'Film samples 2', colours[3]
    z = Read(FilmTwo, 0, 1,1200,1600, 200,500, 330,430, False, title, 0,0,#Pump1, pump2, flat1, flat2, peak1, peak2, spotsize_a and spotsize_b, sen_adj and the title
                Newmodel, colour,1.1, time_PL) 
    
    title, colour = 'Crystal samples 1', colours[0]
    z = Read(CrystalOne,0, 80,1200,1600, 300,600, 300,300, False, title, 0,0,#Pump1, pump2, flat1, flat2, peak1, peak2, spotsize_a and spotsize_b, sen_adj and the title
              Newmodel, colour,1, time_PL) 
    
    title, colour =  'Crystal samples 2', colours[2]
    z = Read(CrystalTwo, 
              0, 1,1400,1600, 300,450, 130,200, True, title, 13, 15, #Pump1, pump2, flat1, flat2, peak1, peak2, spotsize_a and spotsize_b, sen_adj and the title
              Newmodel, colour, 1.1, time_PL)
    #Graphical adjustment things, titles / axes / font size / ticks / limits
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.legend(frameon = False, ncol = 1, title=str(array[int((Newmodel)-1)]) + ' model')#loc='lower right'
    TTitle = ['Cannot-be-called-no-0-model','Singlet-only','Optical depth','Triplet-included','Free-charge model','singlet-triplet annihilation model']
    # ax.legend(frameon=False, ncol=1, title= str(TTitle[Newmodel])+' model')
    plt.xscale('log'),plt.xlabel('Fluence (μJ cm\u207B\u00B2)' ),plt.ylabel('ηPL/ηPL\u2080 (A.U)')
    plt.ylim(0, 1), plt.xlim(0.1, 1000)
    Fig.savefig('Bigplot.svg',bbox_inches='tight')
    # fig.savefig('v'+ titl + titless + str(Newmodel)+ '.svg',bbox_inches='tight') #Save the plots as png or svg as you desire
    plt.show()
    BigPlot=0
    if TestForTamir == 1:
        TestForTamir = 0

###Other lines of data, useful for the weighted averages model but might not go into final paper
###Removed####

###Alternate plotting section, mostly for SI related plots
#Section for plotting the varying ksta values
# VarySTA = 1
if VarySTA == True:
    VaryTTA, VaryISC, VaryKTS, ErrorPlot, ThreeinOne = 0,0,0,0,0
    ax, Fig = Figload()
    coloursInsideL = coloursInsideL = ['#fef0d9','#fdcc8a','#fc8d59','#e34a33','#b30000'][::-1]
    for l in np.arange(0,0.5, 0.1):
        print(l)
        Newmodel = 5
        ksta2 = kb2*l
        title, colour = 'Crystal 1 (Flat, new)(FC model)', coloursInsideL[int(l*10)]
        z1 = Read(CrystalOne, 
                  0, 80,1200,1600, 300,600, 300,300, False, title, 0,0,#Pump1, pump2, flat1, flat2, peak1, peak2, spotsize_a and spotsize_b, sen_adj and the title
                  Newmodel, colour,1, time_PL)
        ksta2 = 0.1*kb2
    ax.scatter(z1[10], z1[5], color = '#b30000', label = 'crystal samples 1',zorder=1)
    ax.tick_params(axis='both', which='major', labelsize=12)
    # ax.legend(frameon = False, ncol = 1, title=str(array[int((Newmodel)-1)]) + ' model')#loc='lower right'
    # ax.legend(frameon=False, ncol=1, title='$K_{STA}$ multiple')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [5,0,1,2,3,4]
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],frameon=False, ncol=1, title='$K_{STA}$ multiple')
    plt.xscale('log'),plt.xlabel('Fluence (μJ cm\u207B\u00B2)' ),plt.ylabel('ηPL/ηPL\u2080 (A.U)')
    plt.ylim(0, 1), plt.xlim(0.1, 1000)
    Fig.savefig('varying STA.svg',bbox_inches='tight')
    plt.show()
    coloursInsideL = coloursInsideL[::-1]
    VarySTA = 0

#Section for plotting the varying ktta values
# VaryTTA = 1
if VaryTTA == True: 
    VarySTA, VaryISC, VaryKTS, ThreeinOne, ErrorPlot = 0,0,0,0,0
    ax, Fig = Figload()
    coloursInsideL = ['#fef0d9','#fdcc8a','#fc8d59','#ef6548','#e34a33','#b30000']
    Newmodel = 3
    for l in np.arange(0,1.2, 0.2):
        print(l)
        Placeholder = ktta
        ktta = ktta*l
        title, colour = 'Crystal 1 (Flat, new)(FC model)', coloursInsideL[int(l*5)]
        z1 = Read(CrystalOne, 
                  0, 80,1200,1600, 300,600, 300,300, False, title, 0,0,#Pump1, pump2, flat1, flat2, peak1, peak2, spotsize_a and spotsize_b, sen_adj and the title
                  Newmodel, colour,1, time_PL)
        ktta = Placeholder
    ax.scatter(z1[10], z1[5], color = '#b30000', label = 'crystal samples 1',zorder=1)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    # ax.legend(frameon = False, ncol = 1, title=str(array[int((Newmodel)-1)]) + ' model')#loc='lower right'
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [6,5,4,3,2,1,0]  
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],frameon=False, ncol=1, title='$K_{TTA}$ multiple')
    plt.xscale('log'),plt.xlabel('Fluence (μJ cm\u207B\u00B2)' ),plt.ylabel('ηPL/ηPL\u2080 (A.U)')
    plt.ylim(0, 1), plt.xlim(0.1, 1000)
    Fig.savefig('varying TTA.svg',bbox_inches='tight')
    # # fig.savefig('v'+ titl + titless + str(Newmodel)+ '.png',bbox_inches='tight') #Save the plots as png or svg as you desire
    plt.show()
    VaryTTA=0

# VaryISC = 1
if VaryISC == True:
    plt.rcParams.update({'font.size': 13})
    VarySTA, VaryTTA,VaryKTS, ErrorPlot, ThreeinOne = 0,0,0,0,0
    ax, Fig = Figload()
    coloursInsideL = ['#fef0d9','#fdcc8a','#fc8d59','#ef6548','#e34a33','#b30000']
    Newmodel = 3
    for l in np.arange(0,1.2, 0.2):
        print(l)
        Placeholder = kisc
        kisc = kisc*l
        title, colour = 'Crystal 1 (Flat, new)(FC model)', coloursInsideL[int(l*5)]
        z1 = Read(CrystalOne, 
                  0, 80,1200,1600, 300,600, 300,300, False, title, 0,0,#Pump1, pump2, flat1, flat2, peak1, peak2, spotsize_a and spotsize_b, sen_adj and the title
                  Newmodel, colour,1, time_PL)
        kisc = Placeholder
    ax.scatter(z1[10], z1[5], color = '#b30000', label = 'crystal samples 1',zorder=1)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [6,5,4,3,2,1,0]
    ax.tick_params(axis='both', which='major', labelsize=12)
    # ax.legend(frameon = False, ncol = 1, title=str(array[int((Newmodel)-1)]) + ' model')#loc='lower right'
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],frameon=False, ncol=1, title='$K_{isc}$ multiple') 
    plt.xscale('log'),plt.xlabel('Fluence (μJ cm\u207B\u00B2)' ),plt.ylabel('ηPL/ηPL\u2080 (A.U)')
    plt.ylim(0, 1), plt.xlim(0.1, 1000)
    Fig.savefig('varying ISC.svg',bbox_inches='tight')
    # # fig.savefig('v'+ titl + titless + str(Newmodel)+ '.png',bbox_inches='tight') #Save the plots as png or svg as you desire
    plt.show()
    VaryISC=0

# VaryKTS = 1
if VaryKTS == True:
    VarySTA, VaryTTA, VaryISC, ErrorPlot, ThreeinOne = 0,0,0,0,0
    ax, Fig = Figload()
    coloursInsideL = ['#fef0d9','#fdcc8a','#fc8d59','#ef6548','#e34a33','#b30000']
    Newmodel = 3
    for l in np.arange(0,2.4, 0.4):
        print(l)
        Placeholder = kts
        kts = kts*l
        title, colour = 'Crystal 1 (Flat, new)(FC model)', coloursInsideL[int(l*2.5)]
        z1 = Read(CrystalOne, 
                  0, 80,1200,1600, 300,600, 300,300, False, title, 0,0,#Pump1, pump2, flat1, flat2, peak1, peak2, spotsize_a and spotsize_b, sen_adj and the title
                  Newmodel, colour,1, time_PL)
        kts = Placeholder
    ax.scatter(z1[10], z1[5], color = '#b30000', label = 'crystal samples 1',zorder=1)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [6,5,4,3,2,1,0]
    ax.tick_params(axis='both', which='major', labelsize=12)
    # ax.legend(frameon = False, ncol = 1, title=str(array[int((Newmodel)-1)]) + ' model')#loc='lower right'
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],frameon=False, ncol=1, title='$K_{ts}$ multiple') 
    plt.xscale('log'),plt.xlabel('Fluence (μJ cm\u207B\u00B2)' ),plt.ylabel('ηPL/ηPL\u2080 (A.U)')
    plt.ylim(0, 1), plt.xlim(0.1, 1000)
    Fig.savefig('varying ts.svg',bbox_inches='tight')
    # # fig.savefig('v'+ titl + titless + str(Newmodel)+ '.png',bbox_inches='tight') #Save the plots as png or svg as you desire
    plt.show()
    VaryKTS=0

# ThreeinOne = 1
if ThreeinOne == True:
    Newmodel = 0
    VaryKTS, VaryISC, VaryTTA, VarySTA, ErrorPlot, BigPlot = 0,0,0,0,0,0
    ax, Fig = Figload()
    title, colour = 'Crystal 1 (Flat, new)(FC model)', colours
    z1 = Read(CrystalOne, 
              0, 80,1200,1600, 300,600, 300,300, False, title, 0,0,#Pump1, pump2, flat1, flat2, peak1, peak2, spotsize_a and spotsize_b, sen_adj and the title
              Newmodel, colour,1, time_PL)
    ax.scatter(z1[10], z1[5], color = '#b30000', label = 'Crystal samples 1',zorder=1)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3,0,1,2]
    ax.tick_params(axis='both', which='major', labelsize=13)
    # ax.legend(frameon = False, ncol = 1, title='70 nm fitting')#loc='lower right'
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],frameon=False, ncol=1, title='70 nm fitting') 
    plt.xscale('log'),plt.xlabel('Fluence (μJ cm\u207B\u00B2)' ),plt.ylabel('ηPL/ηPL\u2080 (A.U)')
    plt.ylim(0, 0.75), plt.xlim(0.2, 500)
    Fig.savefig('ThreeinOne.svg',bbox_inches='tight')
    plt.show()
    ThreeinOne = 0

ErrorPlot = 1
if ErrorPlot == 1:
    ax, Fig = Figload()
    VaryTTA,VaryISC, VaryKTS, ThreeinOne, VarySTA = 0,0,0,0,0
    Newmodel = 1
    title, colour = 'Crystal samples 1', colours[0]
    z1 = Read(CrystalOne, 
              0, 80,1200,1600, 300,600, 300,300, False, title, 0,0,#Pump1, pump2, flat1, flat2, peak1, peak2, spotsize_a and spotsize_b, sen_adj and the title
              Newmodel, colour,1, time_PL)
    title, colour =  'Crystal samples 2', colours[2]
    z = Read(CrystalTwo, 
              0, 1,1400,1600, 300,500, 130,200, True, title, 13, 15, #Pump1, pump2, flat1, flat2, peak1, peak2, spotsize_a and spotsize_b, sen_adj and the title
              Newmodel, colour, 1.15, time_PL)
    title, colour = 'Film samples 1', colours[1]
    z = Read(FilmOne, 
              200, 230,1400,1600, 500,800, 150, 150, True, title, 26, 28,#Pump1, pump2, flat1, flat2, peak1, peak2, spotsize_a and spotsize_b, sen_adj and the title
              Newmodel, colour, 1.05, time_PL) 
    title, colour =  'Film samples 2', colours[3]
    z = Read(FilmTwo, 
                0, 1,1200,1600, 200,500, 330,430, False, title, 0,0,#Pump1, pump2, flat1, flat2, peak1, peak2, spotsize_a and spotsize_b, sen_adj and the title
                Newmodel, colour,1.1, time_PL) 
    
    plt.ylim(0, 3), plt.xlim(10,150), plt.ylabel('Calculated error (A.U)'), plt.xlabel('Diffusion lengths (nm)')
    ax.legend(frameon = True, ncol = 1)#loc='lower right'
    Fig.savefig('ErrorAnalysis.svg',bbox_inches='tight')
    plt.show()
    ErrorPlot = 0

# TCSPCPlot = 1
# ModelTrue =1
if TCSPCPlot ==1 or ModelTrue==1:
    ax, Fig = Figload()
    colours = ['#b30000','#e34a33','#810f7c','#8856a7']
    s=20
    x = np.genfromtxt(TCSPCData, delimiter=',')
    X=x[0:, 1:]
    xlim1, xlim2 = int(1460/64), int(10000/64) #1216 is where rise begins?
    NewTime = ((x[:,0][xlim1:xlim2])/1000)-1.608
    n=0
    params = MDL(multi_exp).make_params(A1=4, tau1=1, A2=1, tau2=3, C=0.016)
    title = ['Crystal samples 1', 'Crystal samples 2','Amorphous film']
    for y in np.transpose(X):
        plt.scatter(NewTime, y[xlim1:xlim2], s=s, color = colours[n], label = title[n])
        if ModelTrue==1:            
            varr= MDL(multi_exp).fit(y[xlim1:xlim2], params, t=NewTime) 
            TauWeighted = (varr.params['A1'].value*varr.params['tau1'].value+varr.params['A2'].value*varr.params['tau2'].value)/(varr.params['A1'].value+varr.params['A2'].value) 
            plt.plot(NewTime, (MDL(multi_exp).fit(y[xlim1:xlim2], params, t=NewTime)).best_fit, color = '#000000', label=(f' weighted tau {float("%.3g" % TauWeighted)}'))
            # plt.fill_between(NewTime, varr.best_fit-varr.eval_uncertainty(t=NewTime),          ####Adds in black error bars around the fits 
            #                   varr.best_fit+varr.eval_uncertainty(t=NewTime), color='#000000')
        n=n+1
        # if n==2:
            # plt.plot(NewTime, (MDL(multi_exp).fit(y[xlim1:xlim2], params, t=NewTime)).best_fit, color = '#000000', label = 'fits')#label=(f' weighted tau {float("%.3g" % TauWeighted)}'))
    ax.tick_params(axis='both', which='major', labelsize=13)
    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [3,0,1,2]
    # ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],frameon=False, ncol=1)#, title='70 nm fitting') 
    ax.legend(frameon = False, ncol = 1)#loc='lower right'
    plt.yscale('log'),plt.xlabel('Time (ns)',fontsize=13),plt.ylabel('Normalised counts',fontsize=13)
    plt.xlim(0,7), plt.ylim(0.02, 1)
    Fig.savefig('TCSPC data presentable format' + '.svg',bbox_inches='tight')
    plt.show()    
    result = (MDL(multi_exp).fit(y[xlim1:xlim2], params, t=NewTime))
    x=result.fit_report()
    # print(result.fit_report())
    
    if ModelTrue ==1:
        ModelTrue=0
    if TCSPCPlot ==1:
        TCSPCPlot = 0

##U is what's used to change the colour of the graph, o is what's used to change the model used
##This will give every set of colours and every model for all the sets of data we're using in the paper
# oplot = 1
if oplot==1:
    plt.rcParams.update({'font.size': 11})
    u = [0,1,2,3,4]
    o = [1,2,3,4,5]
    # u=
    # o=
    for v in o:
        Newmodel = v
        print(v)
        for x in u:
            ax, Fig = Figload()
            SavePlot(x, titles, titless, Newmodel, v)
oplot=0

# Ms=1
if Ms==1:
    ## Note! These are extremely strange fits, I'm not sure why I'm fitting them anyways as I only want something to guide the eye to the general trend of the decay
    ax, Fig = Figload()
    MS(CrystalDecay, 'Film',  '#810f7c', 30, False,1)
    MS(FilmDecay, 'Crystal', '#b30000', 1, False,1)
    plt.show()
    ax, Fig = Figload()
    MS(CrystalDecay, 'Film, zero-one norm',  '#810f7c',0.3, True,1)
    MS(FilmDecay, 'Crystal, zero-one norm', '#b30000', 10, True, 10)
    plt.ylim(0,1)
    plt.show()
    MS=0

# ModelTrue=1
# TCSPCIntensity = 1
if TCSPCIntensity==1:
    ax, Fig = Figload()
    colours = ['#b30000','#e34a33']
    s=5
    x = np.genfromtxt(TCSPCIntensityFile, delimiter=',', skip_header = 1)
    n=0
    X=x[0:, 1:]
    params = MDL(multi_exp).make_params(A1=1, tau1=1000, A2=1, tau2=1000, C=0) #Leftover from TCSPC fitting section, unsure if needed
    Times = ((np.transpose(x)[0])/1000) #Sets the units to ns and the 0 position to be the peak counts
    title = ['Low fluence', 'High fluence'] 
    for y in np.transpose(X):
            plt.scatter(Times, y, s=s, color = colours[n], label = title[n],zorder=1-n)
            if ModelTrue==1:            
                varr= MDL(multi_exp).fit(y, params, t=Times) 
                TauWeighted = (varr.params['A1'].value*varr.params['tau1'].value+varr.params['A2'].value*varr.params['tau2'].value)/(varr.params['A1'].value+varr.params['A2'].value) 
                plt.plot(Times, (MDL(multi_exp).fit(y, params, t=Times)).best_fit, color = 'black', label=(f' weighted tau {float("%.3g" % TauWeighted)}'))
            n=n+1
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(frameon = False, ncol = 1, fontsize = 16, markerscale = 2)#loc='lower right'
    plt.yscale('log'),plt.xlabel('Time (ns)',fontsize=16),plt.ylabel('Normalised counts',fontsize=16)
    plt.xlim(0.1,6), plt.ylim(0.005, 1)
    # plt.xscale('log')
    Fig.savefig('FLuencedependenceinY6xtal' + '.svg',bbox_inches='tight')
    plt.show()
    if ModelTrue ==1:
        ModelTrue=0
    if TCSPCIntensity ==1:
        TCSPCIntensity = 0

# Kinetics = 1
if Kinetics ==1:
        fluence_sim_uj_cm = np.array([0.1,1,10,100])
        fluence_sim = fluence_sim_uj_cm * 1e-6 * 1e4
        ax, Fig = Figload()
        for i in np.arange(0, len(fluence_sim),1):
            N_0 = 1e-6*fluence_sim[i] / (h * c / wavelength)
            P0 = [0,0,0]
            # Compute eta_rho
            eta_rho = []
            eta_rho_i = []
            rho_0_i = N_0* (1 - np.exp(-alpha * dx)) / dx
            sol = solve_ivp(dP_dt6, [time_PL[0], time_PL[-1]], P0, t_eval=time_PL, args=(rho_0_i,), method='BDF')
            Ps = sol.y.T
            S_t = Ps[:, 0]
            T_t = Ps[:, 1]
            ax.plot(time_PL*1e12, S_t/rho_0_i,color = pcolour[i],label = f'Singlet ({float("%.3g" % fluence_sim_uj_cm[i])} ujcm\u00b2)')
            ax.plot(time_PL*1e12, T_t/rho_0_i,color = ocolour[i],label = f'Triplet ({float("%.3g" % fluence_sim_uj_cm[i])} ujcm\u00b2)')
            handles,labels = ax.get_legend_handles_labels()
            # ax.legend(handles, labels, loc='upper right')
            ax.legend(frameon = True, ncol = 1, fontsize = 10, markerscale = 2,loc='upper right')
            ax.set_xlabel('Time (ps)')
            ax.set_ylabel('Normalized population')
            # ax.set_ylim(1e-3,1)
            plt.xscale('log')
            ax.set_xlim(1e-1,1e7)
        Fig.savefig('SingletTripletKinetics.svg',bbox_inches='tight')
        plt.show()
        
        ax, Fig = Figload()
        for i in np.arange(0, len(fluence_sim),1):
            N_0 = 1e-6*fluence_sim[i] / (h * c / wavelength)
            P0 = [0,0,0]
            eta_rho = []
            eta_rho_i = []
            rho_0_i = N_0* (1 - np.exp(-alpha * dx)) / dx
            sol = solve_ivp(dP_dt6, [time_PL[0], time_PL[-1]], P0, t_eval=time_PL, args=(rho_0_i,), method='BDF')
            Ps = sol.y.T
            S_t = Ps[:, 0]
            T_t = Ps[:, 1]
            ax.plot(time_PL*1e12, S_t/rho_0_i,color = pcolour[i],label = f'Singlet ({float("%.3g" % fluence_sim_uj_cm[i])} ujcm\u00b2)')
            ax.plot(time_PL*1e12, T_t/rho_0_i,color = ocolour[i],label = f'Triplet ({float("%.3g" % fluence_sim_uj_cm[i])} ujcm\u00b2)')
            handles,labels = ax.get_legend_handles_labels()
            ax.legend(frameon = True, ncol = 1, fontsize = 10, markerscale = 2,loc='upper right')
            ax.set_xlabel('Time (ps)')
            ax.set_ylabel('Normalized population')
            ax.set_ylim(1e-3,1.1)
            plt.yscale('log')
            plt.xscale('log')
            ax.set_xlim(1e-1,1e7)
        plt.show()
        Fig.savefig('SingletTripletKineticsYNorm.svg',bbox_inches='tight')
        Kinetics = 0

# Tpopplot = 1
if Tpopplot ==1:
        tplot = []
        ax, Fig = Figload()
        # fluence_sim_uj_cm = np.array([0.001,0.01,0.1,1,10,100, 1000])
        fluence_sim_uj_cm = np.logspace(-3,3,num=50)
        fluence_sim = fluence_sim_uj_cm * 1e-6 * 1e4
        for i in np.arange(0, len(fluence_sim),1):
            N_0 = 1e-6*fluence_sim[i] / (h * c / wavelength)
            
            P0 = [0,0,0]
            # Compute eta_rho
            eta_rho = []
             
            eta_rho_i = []
             
            rho_0_i = N_0* (1 - np.exp(-alpha * dx)) / dx
            sol = solve_ivp(dP_dt6, [time_PL[0], time_PL[-1]], P0, t_eval=time_PL, args=(rho_0_i,), method='BDF')
            Ps = sol.y.T
            S_t = Ps[:, 0]
            T_t = Ps[:, 1]
            tplot = np.append(tplot,T_t/rho_0_i)
            sys.stdout.write('.'); sys.stdout.flush();
            
        ttttt= np.transpose(np.reshape(tplot,(len(fluence_sim),len(time_PL))))
        y = fluence_sim  
        x= [max(ttttt[:,i]) for i in np.arange(0, len(np.transpose(ttttt)),1)]
            
        plt.scatter(y,x,c=x, cmap='plasma_r', s=0.2)
        plt.xscale('log')
        plt.xlabel('Fluence in ujcm\u00b2')
        plt.ylabel('Maximum of normalized triplet population')
        # plt.colorbar() #Colorbar adds nothing to this plot.
        Fig.savefig('Tripletpopulation.svg',bbox_inches='tight')
        plt.show()
        Tpopplot = 0

print(''
      'All done :)')
print("--- %s seconds ---" % (time.time() - start_time))



