import re
import random
import pandas as pd

from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler


def read_dataset():
    df = pd.read_csv('drug_consumption.csv')        # read from file
    df = df[df.semeron == 'CL0']                    # drop 'Semeron' admitters
    df = df.drop(labels=['id', 'semeron'], axis=1)  # drop id, semeron columns
    return df


def map_to_actual_values(df):
    # map normalized age ranges to actual age ranges
    age_vals = [-1, -0.5, 0, 0.5, 1.5, 2, 3]
    ages = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    df['age'] = pd.cut(df['age'], age_vals, labels=ages)

    # map normalized gender values to actual genders
    gender_vals = [-1, 0, 1]
    genders = ['M', 'F']
    df['gender'] = pd.cut(df['gender'], gender_vals, labels=genders)

    # map normalized education values to actual education values
    education_vals = [-2.5, -2, -1, -0.1, 0.5, 1.5, 2]
    education = ['elementary', 'high_school',
                 'college', 'bachelor', 'masters', 'doctorate']
    df['education'] = pd.cut(df['education'], education_vals, labels=education)

    # map normalized country values to actual countries
    country_vals = [-1, -0.5, -0.3, -0.1, 0.2, 0.23, 0.25, 1]
    countries = ['usa', 'new_zealand', 'other',
                 'australia', 'ireland', 'canada', 'uk']
    df['country'] = pd.cut(df['country'], country_vals, labels=countries)

    # map normalized ethnicity values to actual ethnicities
    ethnic_vals = [-1.5, -1, -0.4, -0.3, -0.2, 0.12, 1, 2]
    ethnicities = ['black', 'asian', 'white', 'white_black',
                   'other', 'white_asian', 'black_asian']
    df['ethnicity'] = pd.cut(df['ethnicity'], ethnic_vals, labels=ethnicities)

    # convert normalized 12-60 range NEO-FFI values to custom 1-12 scale
    # neuroticism
    neurotic_vals = [-3.5, -2.5, -2, -1.4,
                     -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.7, 2.2, 4]
    neuroticisms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    df['neuroticism'] = pd.cut(
        df['neuroticism'], neurotic_vals, labels=neuroticisms)

    # extraversion
    extravers_vals = [-4, -3.5, -2.5, -2.1,
                      -1.6, -1, -0.5, 0.1, 0.7, 1.3, 2, 3, 4]
    extraversions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    df['extraversion'] = pd.cut(
        df['extraversion'], extravers_vals, labels=extraversions)

    # openness to experience
    open_vals = [-6, -5, -4, -3, -2.5, -1.9,
                 -1.3, -0.8, -0.2, 0.3, 1, 1.7, 3]
    openness = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    df['openness_to_experience'] = pd.cut(
        df['openness_to_experience'], open_vals, labels=openness)

    # agreeableness
    agree_vals = [-3.5, -3.1, -2.8, -2.25, -1.65,
                  -1.1, -0.5, 0, 0.6, 1.3, 2.1, 3.3, 3.5]
    agreeableness = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    df['agreeableness'] = pd.cut(
        df['agreeableness'], agree_vals, labels=agreeableness)

    # conscientiousness
    consc_vals = [-4, -3.5, -3.25, -2.5, -2,
                  -1.6, -1.1, -0.6, -0.1, 0.5, 1.2, 2.1, 3.5]
    conscientiousness = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    df['conscientiousness'] = pd.cut(
        df['conscientiousness'], consc_vals, labels=conscientiousness)

    # convert normalized BIS-11 impulsiveness values to custom 0-10 scale
    impulse_vals = [-4, -3, -2, -1, -0.5, 0, 0.5, 0.75, 1, 1.5, 2, 3]
    impulsiveness = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    df['impulsiveness'] = pd.cut(
        df['impulsiveness'], impulse_vals, labels=impulsiveness)

    # convert normalized ImpSS sensation seeking values to custom 0-10 scale
    ss_vals = [-2.5, -2, -1.5, -1, -0.75, -0.5, 0, 0.25, 0.5, 1, 1.5, 2]
    sensation_seeking = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    df['sensation_seeking'] = pd.cut(
        df['sensation_seeking'], ss_vals, labels=sensation_seeking)

    # mapping substance use time periods to binary values
    # usage more than 10 years ago mapped to false because
    # i feel like it makes the age attribute less valuable
    # for usage 'in last decade' 50% randomness applied
    # because ~2 years ago is not as valid as ~8 years ago

    def substance_map(usage_class):
        if(usage_class in ['CL0', 'CL1']):
            return False
        elif(usage_class == 'CL2'):
            return random.choice([False, True])
        else:
            return True

    df.iloc[:, 12:] = df.iloc[:, 12:].applymap(substance_map)

    return df


def prepare_for_classification(df):
    new_df = df.copy()
    # get original column names
    cols = new_df.columns
    new_df = new_df.apply(LabelEncoder().fit_transform)     # label encode
    vals = new_df.values
    vals = MinMaxScaler().fit_transform(vals)               # minmax scale
    new_df = DataFrame(vals)
    new_df = new_df.apply(round, args=(2,))                 # round
    # reassign column names
    new_df.columns = cols

    new_df.to_csv('classification.csv')

    inputs = new_df.iloc[:, :12].values
    classes = new_df.iloc[:, 12:].values

    (train_inputs, test_inputs, train_classes, test_classes) = train_test_split(
        inputs, classes, train_size=0.7)

    return train_inputs, test_inputs, train_classes, test_classes


def prepare_for_association(df):
    new_df = df.copy()
    # map numerical values to one of three classes
    twelve_scale_bins = [0, 4.5, 9.5, 13]
    ten_scale_bins = [-1, 3.5, 7.5, 11]
    names = ['low', 'medium', 'high']
    new_df['neuroticism'] = pd.cut(
        new_df['neuroticism'], twelve_scale_bins, labels=names)
    new_df['extraversion'] = pd.cut(
        new_df['extraversion'], twelve_scale_bins, labels=names)
    new_df['openness_to_experience'] = pd.cut(
        new_df['openness_to_experience'], twelve_scale_bins, labels=names)
    new_df['agreeableness'] = pd.cut(
        new_df['agreeableness'], twelve_scale_bins, labels=names)
    new_df['conscientiousness'] = pd.cut(
        new_df['conscientiousness'], twelve_scale_bins, labels=names)
    new_df['impulsiveness'] = pd.cut(
        new_df['impulsiveness'], ten_scale_bins, labels=names)
    new_df['sensation_seeking'] = pd.cut(
        new_df['sensation_seeking'], ten_scale_bins, labels=names)

    # one hot encode the dataset
    ohenc = OneHotEncoder(sparse=False)
    original_columns = new_df.columns
    new_df = DataFrame(ohenc.fit_transform(new_df))
    feature_names = ohenc.get_feature_names()
    new_feature_names = []

    # rename attributes from xIndex_Value to Attribute_Value
    for name in feature_names:
        m = re.match('x(\d+)_', name)
        idx = int(m.group(1))
        new_feature_names.append(name.replace(
            f'x{idx}', original_columns[idx]))

    # assign column names and reorder them
    new_df.columns = new_feature_names
    reorder_feature_names = ['age_18-24',
                             'age_25-34',
                             'age_35-44',
                             'age_45-54',
                             'age_55-64',
                             'age_65+',
                             'gender_M',
                             'gender_F',
                             'education_elementary',
                             'education_high_school',
                             'education_college',
                             'education_bachelor',
                             'education_masters',
                             'education_doctorate',
                             'country_canada',
                             'country_usa',
                             'country_new_zealand',
                             'country_australia',
                             'country_ireland',
                             'country_uk',
                             'country_other',
                             'ethnicity_white',
                             'ethnicity_asian',
                             'ethnicity_black',
                             'ethnicity_white_asian',
                             'ethnicity_white_black',
                             'ethnicity_black_asian',
                             'ethnicity_other',
                             'neuroticism_low',
                             'neuroticism_medium',
                             'neuroticism_high',
                             'extraversion_low',
                             'extraversion_medium',
                             'extraversion_high',
                             'openness_to_experience_low',
                             'openness_to_experience_medium',
                             'openness_to_experience_high',
                             'agreeableness_low',
                             'agreeableness_medium',
                             'agreeableness_high',
                             'conscientiousness_low',
                             'conscientiousness_medium',
                             'conscientiousness_high',
                             'impulsiveness_low',
                             'impulsiveness_medium',
                             'impulsiveness_high',
                             'sensation_seeking_low',
                             'sensation_seeking_medium',
                             'sensation_seeking_high',
                             'alcohol_False',
                             'alcohol_True',
                             'amphetamine_False',
                             'amphetamine_True',
                             'amyl_nitrite_False',
                             'amyl_nitrite_True',
                             'benzodiazepine_False',
                             'benzodiazepine_True',
                             'caffeine_False',
                             'caffeine_True',
                             'cannabis_False',
                             'cannabis_True',
                             'chocolate_False',
                             'chocolate_True',
                             'cocaine_False',
                             'cocaine_True',
                             'crack_False',
                             'crack_True',
                             'ecstasy_False',
                             'ecstasy_True',
                             'heroin_False',
                             'heroin_True',
                             'ketamine_False',
                             'ketamine_True',
                             'nps_False',
                             'nps_True',
                             'lsd_False',
                             'lsd_True',
                             'meth_False',
                             'meth_True',
                             'mushrooms_False',
                             'mushrooms_True',
                             'nicotine_False',
                             'nicotine_True',
                             'vsa_False',
                             'vsa_True']
    new_df = new_df[reorder_feature_names]

    new_df.to_csv('association.csv')

    return new_df
