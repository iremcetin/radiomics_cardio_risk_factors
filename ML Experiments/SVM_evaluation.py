# -*- coding: utf-8 -*-
"""
Created on Mar 8 2019
@author: Irem Cetin
email : irem.cetin@upf.edu
"""
################################################################################
# THIS SCRIPT IS FOR ANALYZE THE RISK FACTORS WITH SVMs
# Tested with Python 2.7 and Python 3.5 on Ubuntu Mate Release 16.04.5 LTS (Xenial Xerus) 64-bit
###############################################################################




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
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


def ROC_curve(X, y,setA,label,clf,name,path_to_save):
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
    title='%s_%s_%s'%(setA[0],label,name)
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
    conventional_indices_training_nor=conventional_indices_training_nor.fillna(conventional_indices_training_nor.mean())
    conventional_indices_LV_training = conventional_indices_training_nor.filter(regex=( 'LV'))
    conventional_indices_LV_training =conventional_indices_LV_training.iloc[:,:-1]
#    conventional_indices_LA_training = conventional_indices_training_nor.filter(regex=( 'LA'))
#    conventional_indices_LA_training =conventional_indices_LA_training.iloc[:,:-1]
    conventional_indices_RV_training = conventional_indices_training_nor.filter(regex=('RV'))
    conventional_indices_RV_training =conventional_indices_RV_training.iloc[:,:-1]

    conventional_all_training_nor = pd.concat([conventional_indices_LV_training,conventional_indices_RV_training],axis=1)
    return conventional_all_training_nor

file_feat=open('.../conditions.txt')
conditions=[]
with open(".../conditions.txt") as feat:
    for line in feat:
        f=line.strip()
        f=line.split(",")
        for i in range (len(f)):
            conditions.append(str(f[i]))
            
            
            
os.chdir(".../Risk Factors_new conditions_even_cases/Diabetes_ML/")

### Define Risk factors to analyze


risk_factors =[
#               ['angina'],
#               ['high cholesterol'],
               ['diabetes'],
#               ['asthma'],
#               ['hypertension',\
#               'essential hypertension']
           ]
           
#### List of CVDs in UK Biobank  ##############################################
cvds_classify=[
               'stroke', \
               'transient ischaemic attack (tia)',\
               'peripheral vascular disease',\
               'deep venous thrombosis (dvt)', \
               'heart valve problem/heart murmur',\
               'cardiomyopathy',\
               'atrial fibrillation',\
               'irregular heart beat', \
               'heart/cardiac problem', \
               'raynaud\'s phenomenon/disease',\
               'heart attack/myocardial infarction',\
               'hypertrophic cardiomyopathy (hcm / hocm)',\
               'aortic regurgitation',\
               'aortic stenosis',\
               'aortic aneurysm',\
               'vasculitis',\
               'varicose veins',\
               'mitral valve prolapse',\
               'pericardial problem',\
               'pericardial effusion',\
               'pericarditis',\
               'svt / supraventricular tachycardia',\
               'wegners granulmatosis',\
               'angina',\
               'high cholesterol', \
               'hypertension','essential hypertension',\
               'diabetes',\
               'asthma',\
               'type 1 diabetes'\
               'type 2 diabetes'\
               'heart arrhythmia'\
               'gestational diabetes'\
               'mitral valve prolapse'\
               'mitral regurgitation / incompetence'\
               'mitral valve disease'\
               'pulmonary embolism +/- dvt'\
               'gestational hypertension/pre-eclampsia'\
               'heart failure/pulmonary odema'\
               'cardiomyopathy'\
               'aortic stenosis'\
               'atrial flutter'\
               'hypertrophic cardiomyopathy (hcm / hocm)'\
               'myocarditis'
               
       ]
       
       
       
cvds_samples=[] 
cvd_classifier_acc=[]  
acc_all=[]
models=[]
#### Take the UKBB files ##########################################################################################
#### these 3 files will be check for the cardiovascular diseases
## to find the samples 
main_path_files = '.../UKBB Data Information/Files/'
#1-
conditions=pd.read_csv(main_path_files+'medical_conditions_.csv', low_memory=False)
#2-
history=pd.read_csv(main_path_files+'health_and_medical_history_.csv', low_memory=False)
#3-
outcomes=pd.read_csv(main_path_files+'health_related_outcomes_.csv', low_memory=False)
###

#### Take the conventional clinical indices to make the comparison of the results #####################################
convention =pd.read_csv(main_path_files+'imaging_heart_mri_qmul_oxford_.csv', low_memory=False)
## Get genetics data (if needed)
#genomics = pd.read_csv('genomics_decoded_headings_decoded_data_2017-May-17_1445_r4d.csv', low_memory=False)
#genomics=genomics.fillna(genomics.mean())
#genomics.drop(genomics.select_dtypes(['object']), inplace=True, axis=1)

#### Take the calculated radiomics features
radiomics_ukbb=pd.read_csv('.../cardiac radiomics for Application 2964/1. Radiomics results calculated.csv', low_memory=False)

'''
Define different classifiers
'''
names = [ "Linear SVM_C01", "Linear SVM_C1", "Linear SVM_C10" ,
         "RBF SVM_gamma_01_C0.1","RBF SVM_gamma_01_C1","RBF SVM_gamma_01_C10",
         "RBF SVM_gamma_1_C0.1","RBF SVM_gamma_1_C1","RBF SVM_gamma_1_C10",
         "RBF SVM_gamma_10_C0.1","RBF SVM_gamma_10_C1","RBF SVM_gamma_10_C10",

         ]

classifiers = [
    SVC(kernel="linear", C=0.1, probability =True),
    SVC(kernel="linear", C=1, probability =True),
    SVC(kernel='linear', C=10, probability=True),
    SVC(gamma=0.1, C=0.1, probability =True),
    SVC(gamma=0.1, C=1, probability =True),
    SVC(gamma=0.1, C=10, probability =True),
    SVC(gamma=1, C=0.1, probability =True),
    SVC(gamma=1, C=1, probability =True),
    SVC(gamma=1, C=10, probability =True),
    SVC(gamma=10, C=0.1, probability =True),
    SVC(gamma=10, C=1, probability =True),
    SVC(gamma=10, C=10, probability =True),

   ]
     
cvds_samples_all=[] 
cvds_samples_random_selection=[]
cvd_classifier_acc=[]  
acc_all=[]
cases=[]
models=[]
models_conv=[]
read_samples_path ='.../Risk Factors_new conditions_even_cases/Results-Single_feat_Last/'
path_to_save_roc='.../Risk Factors_new conditions_even_cases/Diabetes_ML/'
for i in range(len(risk_factors)):

    #### Define the set for each risk to analyze
    setA= risk_factors[i]
    rest_cvds_classify = cvds_classify
    rest_risk_factors = list(filter(lambda x: x not in setA,risk_factors ))
    '''
    Get the radiomics features for Normals and risk factors
    '''            
    # Find CVDs in UK Biobank data and add 'normal' cases as a new instance in the list, cvds_classify
#    print('Analyzing %s in UK Biobank...'%(setA))
#    [nor_df, setA_df, rest_cvds_classify] =find_cvds_ukbb(conditions,radiomics_ukbb, rest_cvds_classify, setA)
    nor_df =pd.read_csv(read_samples_path+'normal_%s.csv'%setA[0])
    setA_df =pd.read_csv(read_samples_path+'setA_df_%s.csv'%setA[0])
    nor_conv=pd.read_csv(read_samples_path+'normal_%s_conv.csv'%setA[0])
    setA_conv =pd.read_csv(read_samples_path+'setA_df_%s_conv.csv'%setA[0])

    ##### Count the number of samples for risk factors and Normals to check the number of cases
    cvds_samples_all.append((setA, setA_df.shape[0], nor_df.shape[0]))
    ### Randomly select cases from control group
    label_nor=nor_df.iloc[:,-1]
    nor_df=nor_df.iloc[:,:-1] ### remove labels from the dataframe
    #########################################
    label_setA=setA_df.iloc[:,-1]
    setA_df = setA_df.iloc[:,:-1]
    cvds_samples_random_selection.append((setA, setA_df.shape[0], setA_df, nor_df.shape[0], nor_df))

    
    scaler =MinMaxScaler(feature_range=(-1,1))
    df_all = pd.concat([nor_df,setA_df])
    cases.append((setA,df_all))
    df =df_all.iloc[:,2:]
    Features_scl = scaler.fit_transform(df.iloc[:,1:].values)
    label_all=pd.concat([label_nor, label_setA])
    Labels=label_all.values
    for name, clf in zip(names, classifiers):
        sfs = SFS(clf, # Define feature selector
               k_features=(2,20), # define the number of features or the range 
               forward=True, 
               floating=False, 
               verbose=2,
               scoring='accuracy', 
#              n_jobs=-1,
               cv=10) ## Select the features using cv
### use the training dataset for feature selection 
    
        sfs1 =sfs
        sfs1= sfs1.fit(Features_scl, Labels) 
        fig1 = plot_sfs(sfs1.get_metric_dict(), kind='std_dev')


        plt.title('Radiomics Feature Selection using %s (w. StdDev)'%name)
        plt.grid()
#        plt.show()
        label='radiomics'
        title_name=name.replace(' ', '_')
        plt.savefig('SFS_PLOT_DIABETES_%s_%s.png'%(label,name))
        plt.close()
        Features_scl_selected = sfs1.transform(Features_scl)
    
        models.append((name,clf,sfs1,sfs1.subsets_, sfs1.k_feature_idx_,Features_scl_selected,Labels))
        X=Features_scl_selected
        y=Labels
        ROC_curve(X,y,setA,label,clf,name,path_to_save_roc)
#    #### Do the same with conventional indices
        clf_ = clf
        clf_conv=clf_
        sfs_conv = SFS(clf_conv, # Define feature selector
               k_features=(1,9), # define the number of features or the range 
               forward=True, 
               floating=False, 
               verbose=2,
               scoring='accuracy', 
#              n_jobs=-1,
               cv=10) ## Select the features using cv
#        nor_conv = get_conventional_indices(convention, nor_df)
#        setA_conv = get_conventional_indices(convention, setA_df)
        df_all_conv = pd.concat([nor_conv,setA_conv])
        Features_scl_conv= scaler.fit_transform(df_all_conv)
        sfs1_conv =sfs_conv
        sfs1_conv= sfs1_conv.fit(df_all_conv, label_all) 
       
        #PLOT
        
        fig2 = plot_sfs(sfs1_conv.get_metric_dict(), kind='std_dev')


        plt.title('Feature Selection using %s (w. StdDev)'%name)
        plt.grid()
#        plt.show()
        label_conv='conventional indices'
#        title_name=name.replace(' ', '_')
        plt.savefig('SFS_PLOT_DIABETES_%s_%s.png'%(label_conv,name))
        plt.close()
        Features_scl_selected_conv = sfs1_conv.transform(Features_scl_conv)
    
        models_conv.append((name,clf,sfs1_conv,sfs1_conv.subsets_, sfs1_conv.k_feature_idx_,Features_scl_selected_conv,Labels))
        X_conv=Features_scl_selected_conv
        y=Labels
        ROC_curve(X_conv,y, setA, label_conv,clf,name,path_to_save_roc)
        

     
     
feats_name =[]
ids=[]
feats_name_conv =[]
ids_conv=[]
for name, clf,i in zip(names, classifiers,range(len(models))):    
#    for j in range(len(risk_factors)):
#    sfs = models[i][0]
#    feature_idx = sfs.k_feature_idx_
        setA=risk_factors[0]
        feature_idx = models[i][4]
        feature_idx_conv = models_conv[i][4]
        ids.append(feature_idx)
        ids_conv.append(feature_idx_conv)
        for j in feature_idx:
            feats_name.append((setA,name,df.columns[j+1]))
        for k in feature_idx_conv:
            feats_name_conv.append((setA, name, df_all_conv.columns[k]))
        
        

table_results_diabetes=[]
from numpy import newaxis
from sklearn.model_selection import cross_validate
for name, clf, j in zip(names, classifiers,range(len(models))):    
   
        setA=risk_factors[0]
        feature_idx = models[j][4]
        cv_results_means = np.zeros((len(feature_idx),1))
        cv_results_means_wo = np.zeros((len(feature_idx),1))
        for i in range(len(feature_idx)):
            Features_scl=models[j][5]
            feats_single = Features_scl[:,i]
            feats_single = feats_single[:,newaxis]
            cv_results = cross_validate(clf,feats_single, models[j][6], cv=10)
            cv_results_means[i]= cv_results['test_score'].mean()
    
    
            feats = models[j][5]
            feats_new = np.delete(feats, [i], axis=1)
            cv_results_wo = cross_validate(clf,feats_new, models[j][6], cv=10)
            cv_results_means_wo[i]= cv_results_wo['test_score'].mean()
        
         
'''
OUTPUTS
'''   
for i in range(len(models)):
     print('%s'%models[i][0])
     count_group_features = np.histogram(models[i][4], bins=[0,97,210,678,684,685,686])[0]    
     print('Total shape features: %d\n'%count_group_features[0])
     print('Total intensity features: %d\n'%count_group_features[1])
     print('Total texture features: %d\n'%count_group_features[2])
     print('Total fractal features: %d\n'%count_group_features[3])
     print('Height: %d\n'%count_group_features[4])
     print('Weight: %d\n'%count_group_features[5])
