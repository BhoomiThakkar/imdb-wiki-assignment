import pandas as pd
import matplotlib.pyplot as plt

'''
Generate histogram by age buckets 
'''

def generate_frequency_bins(df, nbins, destination_dir):
    df['age_bins'] = pd.qcut(df['age'], nbins)
    df_age_1=df.groupby(by=['age_bins']).count()
    df_age_1=df_age_1[['age']]
    df_age_1=df_age_1.reset_index()

    x=[i for i in range(len(df_age_1))]
    plt.bar(x=x,height=df_age_1['age'],width=0.2,color='navy')
    plt.xticks(x,df_age_1['age_bins'],rotation='45')
    plt.xlabel('Age Bins')
    plt.ylabel('Number of Observations')
    plt.title('Frequency Distribution By Age')
    plt.savefig(destination_dir+'FrequencyDistribution_'+str(nbins)+'.png')

