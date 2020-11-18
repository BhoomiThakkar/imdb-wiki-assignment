import numpy as np
import pandas as pd
import datetime as date
from scipy.io import loadmat
from dateutil.relativedelta import relativedelta

'''
Extract DOB, Date of photo take, face scores 1 & 2, gender, filepath
'''


# function definition to extract relevant data fields
def extract_data_fields(data_source, source_dir, destination_dir):
    data_mat = source_dir + data_source + '/' + data_source + '.mat'  # imdb/imdb.mat'
    cols = ['age', 'gender', 'filepath', 'face_score1', 'face_score2']
    ages = []
    dobs = []
    paths = []
    genders = []

    loaded_data = loadmat(data_mat)
    del data_mat

    data = loaded_data[data_source]
    photo_taken = data[0][0][1][0]
    full_path = data[0][0][2][0]
    gender_data = data[0][0][3][0]
    face_score1 = data[0][0][6][0]
    face_score2 = data[0][0][7][0]

    print('Loaded {} Observations for {}'.format(len(face_score2),data_source))

    # Extract full image path ..
    for path in full_path:
        paths.append(data_source+'_crop/' + path[0])
    print('\tPath extraction complete ..')

    # Extract Gender Information: 1 == male
    for i in range(len(gender_data)):
        if gender_data[i] == 1:
            genders.append('male')
        else:
            genders.append('female')
    print('\tGender extraction complete ..')

    # Extracting DOB from file Path
    if data_source=='imdb':
        for file in paths:
            dob = file.split('_')[3]
            dob = dob.split('-')  # yyyy-mm-dd
            if len(dob[1]) == 1:
                dob[1] = '0' + str(dob[1])
            if len(dob[2]) == 1:
                dob[2] = '0' + str(dob[2])

            if dob[0] == '0': # not sure about this
                dob[0]='2000'
            if dob[1] == '00': # mm
                dob[1] = '01'
            if dob[2] == '00': # dd
                dob[2] = '01'
            dobs.append('-'.join(dob))
    else:
        for file in paths:
            dob = file.split('_')[2]
            dobs.append(dob)
            if len(dob) != 10:
                print(dob)
    print('\tDOB extraction complete ..')

    # Age
    for i in range(len(dobs)):
        try:
            d1 = date.datetime.strptime(dobs[i], '%Y-%m-%d')
            d2 = date.datetime.strptime(str(photo_taken[i])+'-07-01', '%Y-%m-%d')
            rdelta = relativedelta(d2, d1)
            diff = rdelta.years
        except Exception as ex:
            # print(ex)
            diff=-1

        ages.append(diff)
    print('\tAge extraction complete ..')

    stacked_data = np.vstack((ages, genders, paths, face_score1, face_score2)).T
    dataframe = pd.DataFrame(stacked_data)
    dataframe.columns = cols
    dataframe.to_csv(destination_dir+data_source+'.csv', index=False)

    return dataframe

