import numpy as np
import pandas as pd
import seaborn as sns
from utility import frequency_bins
import matplotlib.pyplot as plt

'''
1. Plot Frequency Distribution for the compiled dataset 
'''

project_dir='/Users/bhoomithakkar/PycharmProjects/chicago_trial/'
source_dir=project_dir+'data/'
destination_dir=project_dir+'plots/'

df = pd.read_csv(source_dir+'compiled_data_score1.csv')

# Task 2 - Frequency Distribution by bins 10, 20
frequency_bins.generate_frequency_bins(df, 10, destination_dir)
frequency_bins.generate_frequency_bins(df, 20, destination_dir)

# Kernel Density Estimates
sns.kdeplot(df['age'])
sns.kdeplot(df['age'],color='navy')
plt.xlabel('Age Distribution')
plt.title('Kernel Density Estimate for Age')
plt.savefig(destination_dir+'KDensity.png')


# Task 3 - Bucket Size for 15-25 yrs age
BucketSize15_25=df[(df['age']>=15)&(df['age']<=25)]
print(len(BucketSize15_25))
print(np.round(len(BucketSize15_25)/len(df)*100,2))
#78572
#17.76


# Task 4 - Bucket Size for 30 year old male
Male_30=df[(df['gender']=='male')&(df['age']==30)]
print(len(Male_30))
print(np.round(len(Male_30)/len(df)*100,3))
#8458
#1.912

# Box Plot
sns.boxplot(y=df['age'])
plt.ylabel('Age')
plt.title('Box Plot for Age')
plt.savefig(destination_dir+'BoxPlot.png')
