import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import plotly.graph_objects as go
import glob
# user written modules
from pandas_datareader import wb
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import iplot


class dataproject:
    def __init__(self):

        '''Initialize:
        data_til01: data frame with the population estimates from 1992 to 2001
        data_til21: data frame with the population estimates from 2002 to 2019
        data_2020: data frame with the population estimates from 2020
        data_2021: data frame with the population estimates from 2021
        regional_fertility: data frame with the fertility rate for each region
        '''
        self.data_til01 = pd.read_csv('Population estimates 1992-2001.csv')
        self.data_til21 = pd.read_csv('Population estimates 2002-2019.csv')
        self.data_2020 = pd.read_csv('Resident_population_2020.csv')
        self.data_2021 = pd.read_csv('Resident_population_2021.csv')
        self.regional_fertility = pd.read_csv('Fertility.csv')

        
    def merge(self):

        '''Merging function:

        Description:
        The function merges the data from self.data_til01 and self.data_til21 using an outer join, 
        ensuring integrity by validating that the keys are unique in both data frames (a one-to-one relationship).

        Returns:
    
        data: a single data frame with the merged data from data_til01 and data_til21
        '''
        data = pd.merge(self.data_til01, self.data_til21 , how='outer',validate='one_to_one') #validate checks the keys are unique in both data frames (a one-to-one relationship), ensuring integrity
        return data
    
    def cleaning(self):

        '''Cleaning functions:

        Description:
        The function cleans the data from the merged data frame. It renames the columns to have the same name for each year 
        It reshapes the data to have the variable 'year' and two new variables: male and female.

        Returns:
    
        pop_italy_wide: a single, cleaned and reshaped, data frame with the merged data from data_til01 and data_til21
        '''

        # We now rename some variables in order not to have numbers as variable names:
        col_dict = {str(i): f'pop{i}' for i in range(1992, 2019 + 1)}
        
        # Merge the datasets
        data_merged = self.merge()
        
        # Rename columns
        data_merged.rename(columns=col_dict, inplace=True)

        # Reshape the dataset to have the variable 'year'
        pop_italy = pd.wide_to_long(data_merged, stubnames='pop', i=['Age', 'Sex'], j='year')
        pop_italy.reset_index(inplace=True)

        # Reshape the dataset again to create two new variables: one for males and one for females.
        pop_italy_wide = pop_italy.pivot(index=['year', 'Age'], columns='Sex', values='pop')
        pop_italy_wide.sort_values(['year', 'Age'], inplace=True)
        pop_italy_wide.reset_index(inplace=True)
        pop_italy_wide.columns.name = None

        return pop_italy_wide

        
    def concat(self):

        '''Concat function:

        Description:
        The `concat` function concatenates multiple datasets and performs some data cleaning operations. 
        
        Returns:
    
        pop_italy: a single cleaned data frame with the concatenated data from data_2020 and data_2021
        '''

        # Set the 'year' column for each dataset
        self.data_2020['year'] = 2020
        self.data_2021['year'] = 2021
        
        # Concatenate the two datasets
        dta = pd.concat([self.data_2020, self.data_2021], ignore_index=True)

        # Rename variables to have the same names as the other dataset for concatenation
        dta.rename(columns={'Total males': 'Males', 'Total females': 'Females'}, inplace=True)
        
        # Call the cleaning method to get the cleaned data
        cleaned_data = self.cleaning()
        
        # Concatenate the cleaned data with the concatenated dataset
        pop_italy = pd.concat([cleaned_data, dta], ignore_index=True)

        return pop_italy

    
    def construction_of_final_dataset(self):

        '''Construction of final dataset function:

        Description:

        The function filters observations, standardises age labels, converts data types, sorts the dataset by 'year' and 'Age',
        creates a new variable with age groups, aggregates population data by age group and year.
        
        Returns:
    
        pop_italy_agg: aggregated and reshaped dataset containing population information by age group and year.
    
        '''

        #we can drop the observation relative to the total  
        pop_italy = self.concat()[self.concat()['Age'] != 'Total']

        #Since 100 or other will be included in the same group of 100. we replace  '100 and other' with 100
        pop_italy['Age'] = pop_italy['Age'].replace('100 and over', '100')

        #Now we can change the column to int
        pop_italy['Age'] = pop_italy['Age'].astype(int)

        #Let's now sort the dataset
        pop_italy.sort_values(['year','Age'], inplace=True)
        pop_italy.reset_index(drop=True, inplace=True)

        # Create a new column with the age group of each observation
        pop_italy['age_group'] = ''

        for i in range(0, 101, 5):
            if i == 100:
                pop_italy.loc[pop_italy['Age'].between(i, i+5), 'age_group'] = '100-100+'
            else:
                pop_italy.loc[pop_italy['Age'].between(i, i+4), 'age_group'] = f'{i}-{i+4}'

        # Summing up all the observations for each age group
        pop_italy_agg = pop_italy.groupby(['age_group', 'year'])[['Males', 'Females','Total']].sum()

        # Resetting index
        pop_italy_agg.reset_index(inplace=True)

        # Splitting age_group into lower_bound and upper_bound in order to sort the age groups correctly
        pop_italy_agg[['lower_bound', 'upper_bound']] = pop_italy_agg['age_group'].str.split('-', expand=True)

        # Sorting by age groups
        pop_italy_agg['lower_bound'] = pop_italy_agg['lower_bound'].astype(int)
        pop_italy_agg.sort_values(['year','lower_bound'], inplace=True)
        pop_italy_agg.reset_index(inplace=True)
        pop_italy_agg.drop(['index', 'lower_bound', 'upper_bound'], axis=1, inplace=True)


        return pop_italy_agg
 
    def insert_dot(self, value):
        '''Insert dot function:

        Description:

        The function inserts a dot as decimal separation. This function has been created because the dataset in the ISTAT website contains the fertility rate without any decimal separation.

        Args:

        value: the column to be converted

        Returns:

        value: the column with the dot as decimal separation
    
        '''
        if len(str(value)) == 2:
            return '0.' + str(value)
        return str(value)[:1] + '.' + str(value)[1:]

    def fertility(self):

        '''Fertility function:

        Description:

        The function keeps only variables of interest, It applies the function 'insert_dot' to the 'Total fertility rate' column and renames the columns.

        Returns:

        fertility: a single data frame with the fertility rate for each region
        '''

        keep_vars = ['Territory', 'Total fertility rate', 'Event year']
        fertility = self.regional_fertility[keep_vars].copy()  

        # Apply the function to the 'Total fertility rate' column
        fertility['Total fertility rate'] = fertility['Total fertility rate'].apply(lambda x: self.insert_dot(x))
        fertility['Total fertility rate'] = fertility['Total fertility rate'].astype(float)

        # Rename the columns
        fertility.rename(columns={'Territory': 'region', 'Total fertility rate': 'total_fertility_rate', 'Event year': 'year'}, inplace=True)

        return fertility

    
    def maps_interactive(self):
        '''Maps interactive function:

        Description:

        This function generates an interactive choropleth map visualizing the total fertility rate (TFR) in Italian regions over the years 1992 to 2021.

        Args:

        regional_fertility: data frame with the fertility rate for each region

        Returns:
        
        choromap2: an interactive choropleth map visualizing the total fertility rate (TFR) in Italian regions over the years 1992 to 2021.
        '''
        # Raw URL to the GeoJSON file
        geojson_url = 'https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_regions.geojson'
        year = 1992

        metric = 'total_fertility_rate'
        data_slider = []

        # Setting a loop to generate all the traces we will use in our Graph Object
        for i in range(year, 2022, 1):
            df_seg = self.fertility()[self.fertility()['year'] == i]
        
            data = dict(type='choropleth',
                locations=df_seg['region'],
                locationmode='geojson-id',
                geojson=geojson_url,
                featureidkey='properties.reg_name',
                colorscale='tealrose',
                zmax=int(self.fertility()[metric].astype('float64').max() + 0.5),  # color bar max
                zmin=int(self.fertility()[metric].astype('float64').min() + 0.5),  # color bar min
                z=df_seg[metric].astype(float),
                hoverinfo='location',
                colorbar=dict(title='TFR'),
                visible=False  # Set all traces to be invisible initially
            )
            
            data_slider.append(data)

        # Make the first trace visible
        data_slider[0]['visible'] = True

        steps = []

        for i in range(len(data_slider)):
            step = dict(method='restyle',
                        args=['visible', [False] * len(data_slider)],
                        label='Year {}'.format(year + 1*i))
            step['args'][1][i] = True
            steps.append(step)

        slider = [dict(active=0,
                    pad={"t": 1},
                    steps=steps)]

        layout = dict(title='Total Fertility Rate in Italian Regions',
                    geo=dict(resolution=50,
                            scope='europe',
                            lonaxis_range=[5, 20],  # Adjusted longitude range for Italy
                            lataxis_range=[35, 48],  # Adjusted latitude range for Italy
                            projection=dict(type='mercator'),
                            showland=False),  # Eliminate background land
                    title_x=0.5,
                    sliders=slider,
                    width=1000,  # Adjust the width of the map
                    height=800,  # Adjust the height of the map
                    )

        choromap2 = go.Figure(data=data_slider, layout=layout)
        return iplot(choromap2, validate=False)
