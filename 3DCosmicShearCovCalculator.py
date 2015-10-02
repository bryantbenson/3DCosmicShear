import numpy as NP
import scipy
import pandas as pd 
import time
import matplotlib.pyplot as PLT
import CSTools as CST
import math
from scipy import special


def CosmicShearCovCalc(Galaxy_input, pz_input, output_file_name, verbose):
    '''
    #Galaxy_input = catalog to be used for calculations, should be in .csv format with data headers including IDNumber, RA, Dec, E1, E2, and DE, to be imported by pandas
    #pz_input = catalog of galaxy photo z probabilities, should be in .csv format with data headers including IDNumber, pxx, pyy, ..., to be imported by pandas
    #output_file_name = the name of the file you wish to output too
    #Equations in this code come primarily from Kitching et. al. 2015 "3D Cosmic Shear: Cosmology from CFHTLenS"
    '''
    
    #Define Column headers to be used in catalog
    RA = 'RA'
    Dec = 'Dec'
    p_z = 'p_z'
    E1 = 'E1'
    E2 = 'E2'
    DE = 'DE'
    
    #Define variables for later use
    Cat_file = Galaxy_input
    pz_file = pz_input
    outfile = '/afs/sapphire.physics.ucdavis.edu/home/bbenson/Data/CosmicShear/DLS/'+output_file_name +'.csv'
    
    deg2rad = NP.pi/180.0
    h_scale=0.7
    Om=0.3
    Ol=0.7
    Or=0.0
    
    k_min = 0.0
    k_max = 5000.0
    k_step = 10.0
    l_min = 0.0
    l_max = 5000.0
    l_step = 10.0
    
    #Import Catalogs
    if verbose == True:
        print 'Importing Data'
    Cat = pd.read_csv(Cat_file,sep=',',na_values='null')
    N_gal = NP.shape(Cat)[0]
    pz_Cat = pd.read_csv(pz_file,sep=',',na_values='null')
    N_pz = NP.shape(pz_Cat)[1]
    if verbose == True:
        print 'Data imported'
           
       
    #Make a list of k and l points to be used
    k = NP.arange(k_min, k_max, k_step)
    l = NP.arange(l_min, l_max, l_step)
    l_x = NP.array((0.0, 0.1, 0.2, 0.3, 0.4, 0.43588989, 0.5, 0.6, 0.7, 0.71414284, 0.8, 0.8660254, 0.9, 0.91651514, 0.9539392, 0.9797959, 0.99498744, 1.0), ndmin=2)
    l_y = NP.array((1.0, 0.99498744, 0.9797959, 0.9539392, 0.91651514, 0.9, 0.8660254, 0.8, 0.71414284, 0.7, 0.6, 0.5, 0.43588989, 0.4, 0.3, 0.2, 0.1, 0.0), ndmin=2)
    
    N_k = NP.shape(k)[0]
    N_l = NP.shape(l)[0]  
    N_l_s = NP.shape(l_x)[1]
    
    #Calculate radial distance from best fit photo-z for each galaxy
    r_z =((Cat[p_z]+1)**2-1)/(((Cat[p_z]+1)**2+1)*(100*h_scale))
    
    #Convert coordinates to tangent-plane coords, then make them into a grid for fast calculations later
    theta_x, theta_y = CST.tanplanecoords(Cat[RA], Cat[Dec])
    theta_x_grid = NP.repeat(NP.array(theta_x, ndmin=2).T, N_l_s, axis=1)
    theta_y_grid = NP.repeat(NP.array(theta_y, ndmin=2).T, N_l_s, axis=1)
    
    #Calculate transformed gamma and separate into E and B modes
    ###Need to add a weight function based on photo z prob. data, which is not being used yet (CFHTLS currently doesn't weight)
    if verbose == True:
        print 'Performing spherical transform of ellipticities and calculating E and B modes'
    
    TempArray = NP.zeros((N_l_s,8))
    DataArray = NP.zeros((1,8))
    
    for a in range(N_k):
        for b in range(N_l):
            D1 = ((l_y*l[b])**2-(l_x*l[b])**2)/2.0
            D2 = -(l_x*l[b])*(l_y*l[b])
            Cos = NP.cos((l_x*l[b])*theta_x_grid+(l_y*l[b])*theta_y_grid)
            Sin = NP.sin((l_x*l[b])*theta_x_grid+(l_y*l[b])*theta_y_grid)
            J = NP.sqrt(NP.pi/(2*k[a]*r_z))*special.jn(l[b]+0.5,k[a]*r_z)
            J_grid = NP.repeat(NP.array(J, ndmin=2).T, N_l_s, axis=1)
    
            Re_g_1 = NP.sum(NP.array(Cat[E1], ndmin=2).T*J_grid*Cos, axis=0)
            Im_g_1 = -NP.sum(NP.array(Cat[E1], ndmin=2).T*J_grid*Sin, axis=0)
            Re_g_2 = NP.sum(NP.array(Cat[E2], ndmin=2).T*J_grid*Sin, axis=0)
            Im_g_2 = NP.sum(NP.array(Cat[E2], ndmin=2).T*J_grid*Cos, axis=0)
    
            Re_g_E = D1/(D1**2+D2**2)*(D1*(Re_g_1+Re_g_2)+D2*(Im_g_1+Im_g_2))
            Im_g_E = D2/(D1**2+D2**2)*(D1*(Re_g_1+Re_g_2)+D2*(Im_g_1+Im_g_2))
            Re_g_B = D1/(D1**2+D2**2)*(-D2*(Re_g_1+Re_g_2)+D1*(Im_g_1+Im_g_2))
            Im_g_B = D2/(D1**2+D2**2)*(-D2*(Re_g_1+Re_g_2)+D1*(Im_g_1+Im_g_2))
    
            TempArray[:,0] = k[a]
            TempArray[:,1] = l[b]
            TempArray[:,2] = l_x
            TempArray[:,3] = l_y
            TempArray[:,4] = Re_g_E
            TempArray[:,5] = Im_g_E
            TempArray[:,6] = Re_g_B
            TempArray[:,7] = Im_g_B
            DataArray = NP.append(DataArray, TempArray, axis=0)
    
    DataArray = NP.delete(DataArray, 0, 0)
    
    #Calculate covariance terms
    if verbose == True:
        print 'Calculating covariance terms'
        
    Cov_Array = NP.zeros((1,5))
    TempArray = NP.zeros((1,5))
    
    for a in range(N_l):
        mask_l = DataArray[:,1] == l[a]
        g_l_slice = DataArray[mask_l]
        for b in range(N_k):
            mask_k1 = g_l_slice[:,0] == k[b]
            for c in range(N_k):
                mask_k2 = g_l_slice[:,0] == k[c]
                
                Re_g_l_k1 = NP.repeat(NP.array(g_l_slice[:,4][mask_k1], ndmin=2), N_l_s, axis=0)
                Re_g_l_k2 = NP.repeat(NP.array(g_l_slice[:,4][mask_k2], ndmin=2).T, N_l_s, axis=1)
                Im_g_l_k1 = NP.repeat(NP.array(g_l_slice[:,5][mask_k1], ndmin=2), N_l_s, axis=0)
                Im_g_l_k2 = NP.repeat(NP.array(g_l_slice[:,5][mask_k2], ndmin=2).T, N_l_s, axis=1)
                
                TempArray[0,0] = k[b]
                TempArray[0,1] = k[c]
                TempArray[0,2] = l[a]
                TempArray[0,3] = NP.mean(Re_g_l_k1*Re_g_l_k2)
                TempArray[0,4] = NP.mean(Im_g_l_k1*Im_g_l_k2)
                Cov_Array = NP.append(Cov_Array, TempArray, axis=0)
    
    Cov_Array = NP.delete(Cov_Array, 0, 0)
    
   
    #convert data array into a pandas dataframe and save to csv file
    if verbose == True:
        print 'Saving Covariance eq. to output file'
        
    output = pd.DataFrame(Cov_Array, Columns=('k1', 'k2', 'l1', 'l2', 'Re_C', 'Im_C'))
    output.to_csv(outfile)

#define initiation variables for terminal command use
import sys
if len(sys.argv)!=5:
    sys.stderr.write('Usage: blah galaxy_input pz_input output_filename verbose\n')
    print len(sys.argv)
    sys.exit(1)
(junk1, Galaxy_input, pz_input, output_file_name, verbose) = sys.argv
CosmicShearCovCalc(Galaxy_input, pz_input, output_file_name, verbose)