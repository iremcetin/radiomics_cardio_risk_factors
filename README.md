# Radiomics Signatures of Cardiovascular Risk Factors in Cardiac MRI #
### About
CMR radiomics is a novel image quantification technique whereby pixel-level data is analyzed to derive multiple quantifiers of tissue shape and texture. Technological advancements and the availability of high computational power has allowed deployment of machine learning (ML) methods with radiomics features to discriminate disease or predict outcomes. A distinct advantage of radiomics modeling over unsupervised algorithms is the potential for explainability through identification of the most defining radiomic features in the model. It is thought that radiomics features correspond to alterations at both the morphological and tissue levels and thus, the most defining features of a particular condition (or its radiomics signature) may provide insights into its pathophysiology. Within oncology, where radiomics is most well-developed, the incremental value of radiomics models for diagnosis and prognosis have been widely reported. In cardiology, early studies have shown promising results from CMR radiomics models for discrimination of important conditions such as myocarditis, hypertrophic cardiomyopathy, and ischemic heart disease.

In this work, we assess, the performance of CMR radiomics models for identifying changes in cardiac structure and tissue texture due to cardiovascular risk factors. Five risk factor groups were evaluated from UK Biobank participants: hypertension, diabetes, high cholesterol, current smoker, and previous smoker. Each group was randomly matched with an equal number of healthy comparators (without known cardiovascular disease or risk factors). Radiomics analysis was applied to short axis images of the left and right ventricles at end-diastole and end-systole, yielding a total of 684 features per study. Sequential forward feature selection in combination with machine learning (ML) algorithms (support vector machine, random forest, and logistic regression) were used to build radiomics signatures for each specific risk group. We evaluated the degree of separation achieved by the identified radiomics signatures using area under curve (AUC), receiver operating characteristic (ROC), and statistical testing. Logistic regression with L1-regularization was the optimal ML model. Compared to conventional imaging indices, radiomics signatures improved the discrimination of risk factor vs. healthy subgroups as assessed by AUC metric.

### The proposed radiomics workflow

<img src="https://user-images.githubusercontent.com/26603738/148645802-bca802df-b846-4419-9eec-1d879d1880dc.jpg" width="700" height="550">

This repository contains the source for training the proposed model. To access the paper please refer [here](https://www.frontiersin.org/articles/10.3389/fcvm.2020.591368/full#h1) and please cite as follows if you are using the code in this repository in any manner.
>Cetin Irem, Raisi-Estabragh Zahra, Petersen Steffen E., Napel Sandy, Piechnik Stefan K., Neubauer Stefan, Gonzalez Ballester Miguel A., Camara Oscar, Lekadir Karim. "Radiomics Signatures of Cardiovascular Risk Factors in Cardiac MRI: Results From the UK Biobank.", Frontiers in Cardiovascular Medicine, 2020, https://doi.org/10.3389/fcvm.2020.591368

```@ARTICLE{10.3389/fcvm.2020.591368,
AUTHOR={Cetin, Irem and Raisi-Estabragh, Zahra and Petersen, Steffen E. and Napel, Sandy and Piechnik, Stefan K. and Neubauer, Stefan and Gonzalez Ballester, Miguel A. and Camara, Oscar and Lekadir, Karim},
TITLE={Radiomics Signatures of Cardiovascular Risk Factors in Cardiac MRI: Results From the UK Biobank},      
JOURNAL={Frontiers in Cardiovascular Medicine},      
VOLUME={7},      
PAGES={232},
YEAR={2020},      
URL={https://www.frontiersin.org/article/10.3389/fcvm.2020.591368},       
DOI={10.3389/fcvm.2020.591368},      
ISSN={2297-055X},
}
