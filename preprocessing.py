import random

import pandas as pd


def map_to_actual_values(df):
    # map normalized age ranges to actual age ranges
    age_vals = [-1, -0.8, 0.5, 1.1, 2, 2.6, 3]
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


def read_dataset():
    df = pd.read_csv('drug_consumption.csv')    # read from file
    df = df.drop(labels=['id'], axis=1)         # drop id column
    return df


def main():
    df = read_dataset()
    df = map_to_actual_values(df)
    print(df)


if __name__ == "__main__":
    main()
