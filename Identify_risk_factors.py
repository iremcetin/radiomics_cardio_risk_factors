# -*- coding: utf-8 -*-
"""
Created on Mar 8 2019
@author: Irem Cetin
email : irem.cetin@upf.edu
################################################################################
THIS SCRIPT IS FOR ANALYZING THE RISK FACTORS IN UK BIOBANK 
Tested with Python 2.7 and Python 3.5 on Ubuntu Mate Release 16.04.5 LTS (Xenial Xerus) 64-bit
###############################################################################



################################################################################

"""

'''
IMPORT LIBRARIES
'''
import numpy as np
import pandas as pd
import os
from cvd_ids_in_ukbb_normal_pca import find_cvds_ukbb 
#from analyze_plots_ukbb import *
from sklearn.svm import SVC
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp


def ROC_curve(X, y,setA,label,clf,path_to_save):
    cv=StratifiedKFold(n_splits=10)
    classifier = clf
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i=0
    for train, test in cv.split(X,y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of %s using %s'%(setA[0], label))
    plt.legend(loc="lower right")
#plt.show()
    title='%s_%s'%(setA[0],label)
    title.replace(' ','_')
    plt.savefig(path_to_save+'ROC_%s.png'%title)   
    plt.close()
    
    
    
def find_min_overlap(overlap_angina):
    min_overlap_id = overlap_angina.values.argmin()  
    min_overlap_name =overlap_angina.columns[min_overlap_id] 
    return min_overlap_name

def get_conventional_indices (convention, nor_df_training):
    '''
    Get conventional indices for Normal Training
    '''
    conventional_indices_training_nor = convention.loc[convention['f.eid'].isin(nor_df_training['patient'])]
    conventional_indices_training_nor = conventional_indices_training_nor.set_index('f.eid')
    conventional_indices_training_nor = conventional_indices_training_nor.reindex(index = nor_df_training['patient'])



#conventional_indices.reindex(radiomics_hypertension_cvds_df.index)
    conventional_indices_LV_training = conventional_indices_training_nor.filter(regex=( 'LV'))
    conventional_indices_LV_training =conventional_indices_LV_training.iloc[:,:-1]
#    conventional_indices_LA_training = conventional_indices_training_nor.filter(regex=( 'LA'))
#    conventional_indices_LA_training =conventional_indices_LA_training.iloc[:,:-1]
    conventional_indices_RV_training = conventional_indices_training_nor.filter(regex=('RV'))
    conventional_indices_RV_training =conventional_indices_RV_training.iloc[:,:-1]
#    conventional_all_training_nor = pd.concat([conventional_indices_LV_training,conventional_indices_RV_training,\
#                            conventional_indices_LA_training],axis=1)
    conventional_all_training_nor = pd.concat([conventional_indices_LV_training,conventional_indices_RV_training],axis=1)
    return conventional_all_training_nor

os.chdir(".../Risk Factors_new conditions_even_cases/")


### Define Risk factors to analyze
risk_factors =[
               ['high cholesterol'],
               ['diabetes','type 1 diabetes','type 2 diabetes'],
               ['hypertension',\
               'essential hypertension'],
               ['smoking_current','smoking_previous']
           ]

cvds_samples=[] 
cvd_classifier_acc=[]  
acc_all=[]
models=[]


       
cvds_samples_all=[] 
cvds_samples_random_selection=[]
cvd_classifier_acc=[]  
acc_all=[]
cases=[]
models=[]
models_conv=[]

read_samples_path ='.../Risk Factors_new conditions_even_cases/'
path_to_save_roc='.../Risk Factors_new conditions_even_cases/ROC_curves/'
for i in range(len(risk_factors)):

    setA= risk_factors[i]
    
   '''
    Read the cases for Risk factors and controls
    '''
    nor_df =pd.read_csv(read_samples_path+'normal_random_sample_%s.csv'%setA[0])
    setA_df =pd.read_csv(read_samples_path+'setA_df_all_%s.csv'%setA[0])

    ##### Count the number of samples for Risk factors and Normals to check the number of cases -->To check
    cvds_samples_all.append((setA, setA_df.shape[0], nor_df.shape[0]))
    label_nor=nor_df.iloc[:,-1]
    nor_df=nor_df.iloc[:,:-1] ### remove labels from the dataframe
    label_setA=setA_df.iloc[:,-1]
    setA_df = setA_df.iloc[:,:-1]
    cvds_samples_random_selection.append((setA, setA_df.shape[0], setA_df, nor_df.shape[0], nor_df))
    #### Preprocessing ##############################################################################
    scaler =MinMaxScaler(feature_range=(-1,1))
    df_all = pd.concat([nor_df,setA_df])
    cases.append((setA,df_all))
    df =df_all.iloc[:,2:]
    Features_scl = scaler.fit_transform(df.iloc[:,1:].values)
    label_all=pd.concat([label_nor, label_setA])
    Labels=label_all.values
    clf = SVC(kernel='rbf', 
              decision_function_shape='ovr',
              C=1, gamma=0.1,
              class_weight='balanced',
              probability=True,
              random_state=42)

    sfs = SFS(clf, # Define feature selector
               k_features=(1,100), # define the number of features or the range 
               forward=True, 
               floating=False, 
               verbose=2,
               scoring='accuracy', 
#              n_jobs=-1,
               cv=10) ## Select the features using cv
### use the training dataset for feature selection  ##############################  
    
    sfs1 =sfs
    sfs1= sfs1.fit(Features_scl, Labels) 
    Features_scl_selected = sfs1.transform(Features_scl)
    
    models.append((sfs1,sfs1.subsets_, sfs1.k_feature_idx_,Features_scl_selected,Labels))
    X=Features_scl_selected
    y=Labels
    label='radiomics'
    ROC_curve(X,y,setA,label,clf,path_to_save_roc)
#    #### Do the same with conventional indices
    clf_ = SVC(kernel='rbf', 
              decision_function_shape='ovr',
              C=1, gamma=0.1,
              class_weight='balanced',
              probability=True,
              random_state=42)
    clf_conv=clf_
    sfs_conv = SFS(clf_conv, # Define feature selector
               k_features=(1,9), # define the number of features or the range 
               forward=True, 
               floating=False, 
               verbose=2,
               scoring='accuracy', 
#              n_jobs=-1,
               cv=10) ## Select the features using cv
    nor_conv = get_conventional_indices(convention, nor_df)
    setA_conv = get_conventional_indices(convention, setA_df)
    df_all_conv = pd.concat([nor_conv,setA_conv])
    Features_scl_conv= scaler.fit_transform(df_all_conv)
    sfs1_conv =sfs_conv
    sfs1_conv= sfs1_conv.fit(df_all_conv, label_all) 
    Features_scl_selected_conv = sfs1_conv.transform(Features_scl_conv)
    
    models_conv.append((sfs1_conv,sfs1_conv.subsets_, sfs1_conv.k_feature_idx_,Features_scl_selected_conv,Labels))
    X_conv=Features_scl_selected_conv
    y=Labels
    label_conv='conventional indices'
    ROC_curve(X_conv,y, setA, label_conv,clf_conv,path_to_save_roc)
    
    

name =[]
ids=[]
name_conv =[]
ids_conv=[]
for i in range(len(models)):
#    sfs = models[i][0]
#    feature_idx = sfs.k_feature_idx_
    setA=risk_factors[i]
    feature_idx = models[i][2]
    feature_idx_conv = models_conv[i][2]
    ids.append(feature_idx)
    ids_conv.append(feature_idx_conv)
    for j in feature_idx:
        name.append((setA,df.columns[j+1]))
    for k in feature_idx_conv:
        name_conv.append((setA, df_all_conv.columns[k]))
        
        
clf_svm = SVC(kernel='rbf', 
              decision_function_shape='ovr',
              C=1, gamma=0.1,
              class_weight='balanced',
              probability=True,
              random_state=42)
 
table_results=[]
from numpy import newaxis
from sklearn.model_selection import cross_validate
for j in range(len(models)):
    setA=risk_factors[j]
    feature_idx = models[j][2]
    cv_results_means = np.zeros((len(feature_idx),1))
    cv_results_means_wo = np.zeros((len(feature_idx),1))
    for i in range(len(feature_idx)):
        Features_scl_selected=models[j][3]
        feats_single = Features_scl_selected[:,i]
        feats_single = feats_single[:,newaxis]
        cv_results = cross_validate(clf_svm,feats_single, models[j][4], cv=10)
        cv_results_means[i]= cv_results['test_score'].mean()
    
    
        feats = models[j][3]
        feats_new = np.delete(feats, [i], axis=1)
        cv_results_wo = cross_validate(clf_svm,feats_new, models[j][4], cv=10)
        cv_results_means_wo[i]= cv_results_wo['test_score'].mean()
    table_results.append((setA, cv_results_means, cv_results_means_wo))