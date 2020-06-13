# 1: Insightface and Dex CAGe (Count / Age / Gender) analysis

## a: ImageNet_1A_Insightface_generate.ipynb

### Data file needed

- ```
  df_imagenet=pd.read_csv('df_imagenet_stats.csv') 
  ```
- The ImageNet dataset residing in data_dir='/home/shared/datasets/imagenet/'

## b: ImageNet_1B_Insightface_analyze.ipynb

- ```
  df_imagenet=pd.read_csv('df_imagenet_stats.csv') 
  ```
- The ImageNet dataset residing in data_dir='/home/shared/datasets/imagenet/'

### Data output:

```
df_imagenet.to_csv('df_imagenet_census.csv',index=False)
np.save('age_all.npy',np.array(age_all_dataset))
np.save('age_women.npy',np.array(age_women_dataset))
np.save('age_men.npy',np.array(age_men_dataset))
np.save('age_all_train.npy',np.array(age_all_train))
np.save('age_women_train.npy',np.array(age_women_train))
np.save('age_men_train.npy',np.array(age_men_train))
np.save('age_all_val.npy',np.array(age_all_val))
np.save('age_women_val.npy',np.array(age_women_val))
np.save('age_men_val.npy',np.array(age_men_val))
```

## c: ImageNet_1C_census_audit_dex.ipynb

### Data needed:

Make sure you download the two .json files

- ILSVRC2012_training_age_DEX.json
- ILSVRC2012_training_gender_DEX.json from https://github.com/cdulhanty/ImageNet-Demo-Audit/releases/tag/0.1 and place them in the ~/audit/ folder.

### Data output:

```
'df_audit_age_gender_dex.csv'

```

# 2: NSFW analysis (ImageNet_2_NSFW_analysis.ipynb)

## Data output:

```
df_nsfw.to_csv('df_nsfw.csv')
np.save('nsfw_scores_dataset.npy',np.array(nsfw_scores_dataset))
pkl_file='nsfw_imagenet.pkl'
```

# 3: ImageNet_3_Gender_Dog_musical_instruments_final.ipynb (Gender bias analysis)

## Data inputs:

```
df_dog_analysis=pd.read_csv('df_dog_analysis.csv')
df_dogs_imagenet=pd.read_csv('dogs_imagenet.csv')
```

## Data outputs:

```
df_dog_groups.to_csv('df_dog_groups.csv')
```


# 4: ImageNet_4_Semantics_glove_gen.ipynb (Generating glove embeddings for the 1k classes)


## Data inputs:

df_imagenet_names=pd.read_csv('/content/sample_data/df_imagenet_classes.csv')

## Data outputs

```
df_imagenet_names.to_csv('df_imagenet_renamed.csv',index=False)
df_imagenet_names.to_csv('df_imagenet_names_umap.csv',index=False)
np.save('imagenet_glove_300_umap.npy',embeddings_projected)
np.save('imagenet_glove_300.npy',imagenet_glove_300)
```

# 5 : ImageNet_5C_Semanticity_error.ipynb (Taking the glove embeddings generated and forming )

## Data inputs:
```
df_imagenet_names_umap.csv
df_nsfw.csv
df_comb_umap.csv
```
## Data outputs:
```
df_comb.to_csv('df_comb_umap.csv',index=False)
```

# 6: Class-wise accuracy evaluations

## A: ImageNet_6A_acc_ResNet50.ipynb (Resnet-50 evaluations)



### Data Outputs:
df_acc.to_csv('df_acc_classwise.csv')


## B: ImageNet_6B_acc_NasNet.ipynb (NasNet evaluations)


### Data Outputs:
```
# X_tensors: N X 224 x 224 x 3 tensors for each class 
file_out_X=os.path.join(os.getcwd(),'ImageNet_npy_224_raw',f'X_{sub_direc}_{direc}.npy')
np.save(file_out_X,X_class)

# F_tensors: N x N_f tensors:

file_out_F=os.path.join(os.getcwd(),'ImageNet_npy_224_F_NasNet',f'F_{sub_direc}_{direc}.npy')
np.save(file_out_F,F_class)

# Global matrices for each class:

np.save(f'./mols_NasNet_mobile/Grammian_term_NasNet_mobile_{count_iter}.npy',Grammian_term)
np.save(f'./mols_NasNet_mobile/Proj_term_NasNet_mobile_{count_iter}.npy',Proj_term)

# Global matrices for the MOLS idea:
np.save('Grammian_term_NasNet_mobile.npy',Grammian_term)
np.save('Proj_term_NasNet_mobile.npy',Proj_term)
# Accuracy
df_acc.to_csv('df_acc_classwise_NasNet_mobile.csv')
```


# 7: ImageNet_7_Census_munge.ipynb (To munge together the individual csv outputs from the above notebooks to create one big monolithic 'census' csv)
## Data inputs:
```
'df_insightface_stats.csv': InsightFace analysis results
'df_audit_age_gender_dex.csv': Dex analysis results from teh audit paper
'df_nsfw.csv': NSFW analysis
'df_acc_classwise_resnet50.csv': Classwise acc and preds using the ResNet50 model
'df_acc_classwise_NasNet_mobile.csv': Classwise acc and preds using the NasNetMobile model
'df_imagenet_names_umap.csv': UMAP_2D (Glove-300D (Imagenet class labels))
'df_census_columns_interpretation.csv' : Interpretations of what the 61 output columns stand for.
```


## Data output:

```
df_comb.to_csv('df_census_imagenet_61.csv',index=False)
```


# 8: ImageNet_8_Process_hand_survey_files.ipynb

  ## Data inputs:
  ```
  df_list=pd.read_csv('df_hand_survey.csv')
  df_imagenet=pd.read_csv('df_census_imagenet_61.csv')
  ```
  ## Data outputs:
  ```
  df_filt_final.to_csv('df_hand_survey.csv',index=False)
  ```

# 9: tiny_images_1_index.ipynb (Parse through Tiny Images index and get counts of offensive terms)

	## Data inputs:
		```
		file_name='tiny_index.mat'
		from scipy.io import loadmat
		x = loadmat(file_name)
		```
	## Data outputs:
		```
		df_classes_tiny_images.to_csv('df_classes_tiny_images.csv',index=False)
		```
# 10: tiny_images_2_plots.ipynb (Parse through Tiny Images bin file to vizualize the offensive images)
## Data inputs
```
df_classes_tiny_images_3.csv
BinfileName='tiny_images.bin'
```






 
