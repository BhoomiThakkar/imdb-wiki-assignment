import pandas as pd
from utility import extract_data

'''
1. For IMDB, Wiki images - extract DOB, Date of photo take, face scores 1 & 2, 
   gender, filepath
2. Compile IMDB-Wiki dataset
3. Clean noisy face_score1 and date values
'''

project_dir='/Users/bhoomithakkar/PycharmProjects/chicago_trial/'
source_dir=project_dir
destination_dir=project_dir+'data/'

# Extraction from Wiki meta-data
df1=extract_data.extract_data_fields('wiki', source_dir, destination_dir)
# Extraction from IMDB meta-data
df2=extract_data.extract_data_fields('imdb', source_dir, destination_dir)

# Concatenate wiki-imdb data
df=pd.concat([df1, df2])  # 523051

# Visualising the basic summary of the data
null=df.isna().sum()  # 305158 nans for score 2
description=df.describe().transpose()

df['male']=df['gender'].apply(lambda x: 1 if x=='male' else 0)
df['female']=1-df['male']

df['face_score1']=df['face_score1'].astype('str')
df['face_score2']=df['face_score2'].astype('str')

# Eliminate noisy score values
df_score1=df[df['face_score1'] != '-inf']  # 442,733

# Eliminate noisy age values
df_age=df_score1[(df_score1['age'] >= 0) & (df_score1['age'] <= 100)]  # 442322
df_age=df_age.reset_index()
df_age.drop(['index'], axis=1, inplace=True)

# Checking basic description for data
description_age=df_age.describe().transpose()

df_age.to_csv(destination_dir+'compiled_data_score1.csv',index=False)