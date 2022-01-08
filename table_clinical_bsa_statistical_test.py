# -*- coding: utf-8 -*-
"""
Created on Mar 8 2019
@author: Irem Cetin
email : irem.cetin@upf.edu
################################################################################
THIS SCRIPT IS FOR ANALYZING THE RISK FACTORS IN UK BIOBANK DATA
Tested with Python 2.7 and Python 3.5 on Ubuntu Mate Release 16.04.5 LTS (Xenial Xerus) 64-bit
###############################################################################
Analysis of the radiomics features for each risk factor
SINGLE FEATURE ANALYSIS
* Density plots
* Discriminative power
* P value and T value

################################################################################

"""

'''
IMPORT LIBRARIES
'''
import numpy as np
import pandas as pd
import os
#from cvd_ids_in_ukbb_normal_pca import find_cvds_ukbb 
#from analyze_plots_ukbb import *
from scipy.stats import ttest_ind


def du_bois_bsa(Height, Weight):
     const1 = 0.007184
     power_height = 0.725
     power_weight = 0.425
     bsa = const1 * ( Height**power_height) * (Weight**power_weight)
     return bsa


### Define Risk factors to analyze
print('Taking the list of risk factors')
risk_factors =[
#               ['angina'],
               ['high cholesterol'],
               ['diabetes','type 2 diabetes'],
#               ['asthma'],
               ['hypertension',\
               'essential hypertension'],
               ['smoking_current'],['smoking_previous']
           ]

print('In total there is/are %d risk factors'%len(risk_factors) )     
print(risk_factors)
 
#sample_save = '/home/irem/Desktop/ACDC_Test/Normal Analyze/Risk Factors_new conditions_even_cases/'    
read_samples_path ='C:/Users/iremc/Downloads/Data/'

gender_age=pd.read_csv('C:/Users/iremc/Desktop/ACDC_Test/Methods/UKB_5065_characteristics1.csv', low_memory=False)

'''
Create the set for CVD
'''
cvds_samples_all=[]
table_conv_over_bsa=[]
table_conv_over_bsa_nor =[]
table_gender=[]
table_age=[]
for i in range(len(risk_factors)):
    
    setA= risk_factors[i]
    print('Running for %s'%setA[0])
    
    #            setA =''.join(setA_str)
#    rest_cvds_classify=list(filter(lambda x: x not in setA,cvds_classify ))
#    setA= risk_factors[i]
    
    '''
    Read the cases for Risk factors and controls
    '''
    if (risk_factors[i][0]=='smoking_previous')==1 or (risk_factors[i][0]=='hypertension')==1:
#         nor_df =pd.read_csv(read_samples_path+'normal_%s.csv'%setA[0])
         setA_df =pd.read_csv(read_samples_path+'setA_df_%s.csv'%setA[0])
         setA_conv = pd.read_csv(read_samples_path+'setA_df_%s_conv.csv'%setA[0])
    else:
#        nor_df =pd.read_csv(read_samples_path+'normal_%s.csv'%setA[0])
        setA_df =pd.read_csv(read_samples_path+'setA_df_%s.csv'%setA[0])
#        nor_conv = pd.read_csv(read_samples_path+'normal_%s_conv.csv'%setA[0])
        setA_conv = pd.read_csv(read_samples_path+'setA_df_%s_conv.csv'%setA[0])
        
    Height = setA_df['Height']
    Weight = setA_df['Weight']
    
        
    setA_ids = setA_df['patient']
    gender_age_setA = gender_age.loc[gender_age['f.eid'].isin(setA_ids)]
    gender = gender_age_setA['bio.sex.0.baseline']
    male = gender_age_setA.loc[gender_age_setA['bio.sex.0.baseline']=='Male']
    female = gender_age_setA.loc[gender_age_setA['bio.sex.0.baseline']=='Female']
    percentage_male ="{:.1%}".format(np.float(male.shape[0])/np.float(setA_df.shape[0]))
    percentage_female ="{:.1%}".format(np.float(female.shape[0])/np.float(setA_df.shape[0]))
    table_gender.append((setA,male,female,percentage_male))
    
    age =gender_age_setA['bio.age.when.attended.assessment.centre.0.imaging']
    mean_age = np.mean(age)
    std_age=np.std(age)
    table_age.append((setA, age, mean_age, std_age))
    setA_df =setA_df.iloc[:,2:]
    bsa = du_bois_bsa(Height, Weight)
#
#    ##### Count the number of samples for Risk factors and Normals to check the number of cases -->To check
    cvds_samples_all.append((setA, setA_df.shape[0], setA_ids, setA_conv, Height, Weight, bsa))
    nor_conv = pd.read_csv(read_samples_path+'normal_hypertension_conv.csv')
    nor_df = pd.read_csv(read_samples_path+'normal_hypertension.csv')
    nor_ids = nor_df['patient']

    gender_age_nor = gender_age.loc[gender_age['f.eid'].isin(nor_ids)]
    gender_nor = gender_age_nor['bio.sex.0.baseline']
    male_nor = gender_age_nor.loc[gender_age_nor['bio.sex.0.baseline']=='Male']
    female_nor = gender_age_nor.loc[gender_age_nor['bio.sex.0.baseline']=='Female']
    percentage_male_nor ="{:.1%}".format(np.float(male_nor.shape[0])/np.float(nor_df.shape[0]))
    percentage_female_nor ="{:.1%}".format(np.float(female_nor.shape[0])/np.float(nor_df.shape[0]))
    print('Total number of males:')
    print(male_nor.shape[0])
    print('Male percentage :')
    print(percentage_male_nor)    
    age_nor =gender_age_nor['bio.age.when.attended.assessment.centre.0.imaging']
    mean_age_nor = np.mean(age_nor)
    std_age_nor=np.std(age_nor)
    print('Mean age:')
    print(mean_age_nor)
    print('STD age:')
    print(std_age_nor)

    Height_nor = nor_df['Height']
    Weight_nor =nor_df['Weight']
    bsa_nor = du_bois_bsa(Height_nor, Weight_nor)
    conv_bsa_nor = nor_conv.values / bsa_nor.values.reshape(-1,1)
    conv_bsa = setA_conv.values / bsa.values.reshape(-1,1)
    conv_bsa_df_nor = pd.DataFrame(columns = nor_conv.columns, data = conv_bsa_nor)    
    conv_bsa_df =pd.DataFrame(columns=setA_conv.columns, data=conv_bsa)
    conv_bsa_df['cmr:Analysis:LVEF']=setA_conv['cmr:Analysis:LVEF']
    conv_bsa_df_nor['cmr:Analysis:LVEF']=nor_conv['cmr:Analysis:LVEF']
    conv_bsa_df['cmr:Analysis:RVEF'] = setA_conv['cmr:Analysis:RVEF']
    conv_bsa_df_nor['cmr:Analysis:RVEF']=nor_conv['cmr:Analysis:RVEF']


    conv_bsa_df_mean = conv_bsa_df.mean()
    conv_bsa_df_std = conv_bsa_df.std()
    conv_bsa_df_nor_mean = conv_bsa_df_nor.mean()
    conv_bsa_df_nor_std = conv_bsa_df_nor.std()
    table_conv_over_bsa.append((setA,setA_conv, Height, Weight, bsa, conv_bsa, conv_bsa_df_mean, conv_bsa_df_std ))
    table_conv_over_bsa_nor.append((nor_conv, Height, Weight, bsa, conv_bsa_nor, conv_bsa_df_nor_mean, conv_bsa_df_nor_std))
 
################################################################## STATISTICAL TEST ###########################################
#### take the data risk factors - clinical indices vs. healthy - clinical indices
for j in range(len(table_conv_over_bsa)):
    print('Risk Factor : \n')
    print(table_conv_over_bsa[j][0])
    for i in range(setA_conv.shape[1]):
        seta_conv = table_conv_over_bsa[j][1]
        nor_conv = table_conv_over_bsa_nor[j][0]
        print('T-test %s'%seta_conv.columns[i])
        stat_ttest = ttest_ind(seta_conv.iloc[i], nor_conv.iloc[i],equal_var=False)
        print('t-value \n')
        print(stat_ttest.statistic)
        print('\n')
        print('p-value \n')
        print(stat_ttest.pvalue)
        print('\n')
