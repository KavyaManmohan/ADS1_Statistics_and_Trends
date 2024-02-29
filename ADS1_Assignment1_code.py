# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 02:22:30 2024

@author: LENOVO
"""

# Import
import pandas as pd  # To read the file
import numpy as np  # To the mathematical operation
import matplotlib.pyplot as plt  # To the visualisation

# Function


def read_process_data(file_name, indi_name, countries_name):
    """
    Read and process data from a CSV file.

    Parameters:
    file_name (str): Name of the CSV file.
    indi_name (str): Name of the indicator to extract from the data.
    countries_name (list): List of country names to filter data.

    Returns:
    DataFrame: Original DataFrame after processing.
    DataFrame: Transposed DataFrame after processing.
    """
    # Read CSV file skipping first 3 rows
    df = pd.read_csv(file_name, skiprows=3)

    # Filter data by indicator name and countries
    data = df[(df['Indicator Name'] == indi_name) & (
        df['Country Name'].isin(countries_name))]

    # Drop unnecessary columns
    columns_drop = ['Indicator Name', 'Indicator Code',
                    'Country Code', 'Unnamed: 67']
    df1 = data.drop(columns_drop + [str(year) for year in range(1960, 1990)] +
                    [str(year) for year in range(2021, 2023)],
                    axis=1).reset_index(drop=True)

    # Transpose DataFrame
    t_df1 = df1.transpose()
    t_df1.columns = t_df1.iloc[0]
    t_df1 = t_df1.iloc[1:]
    t_df1['Years'] = pd.to_numeric(t_df1.index)

    # Convert columns to numeric
    for column in t_df1.columns[:-1]:  # Exclude the 'Years' column
        t_df1[column] = pd.to_numeric(
            t_df1[column], errors='coerce').fillna(0).astype(int)

    return df1, t_df1


def lineplot(df, countries):
    """
    Plot a line graph of cereal production over time for specified countries.

    Parameters:
    df (DataFrame): DataFrame containing cereal production data.
    countries (list): List of countries to plot.

    Returns:
    None
    """
    plt.figure(figsize=(7, 4))
    for country in countries:
        plt.plot(df['Years'], df[country], marker='o', label=country)

    plt.title('Cereal Production Line plot Over Time')
    plt.xlabel('Year')
    plt.ylabel('kg per Hectare')
    plt.legend()
    plt.grid(True)
    plt.savefig('lineplot.png', dpi=300)  # Save plot as image
    plt.show()


def barplot(df, countries):
    """
    Plot a grouped bar graph of cereal production for specified countries.

    Parameters:
    df (DataFrame): DataFrame containing cereal production data.
    countries (list): List of countries to plot.

    Returns:
    None
    """
    plt.figure(figsize=(7, 4))
    bar_width = 0.7
    years = df['Years']
    start_year = 1990
    end_year = 2020
    # Define x-axis ticks for every 5 years
    x = np.arange(start_year, end_year + 1, 5)

    for i, country in enumerate(countries):
        plt.bar(x + i * bar_width,
                df[country][(x - start_year)], width=bar_width, label=country)

    plt.xlabel('Years')
    plt.ylabel('kg per Hectare')
    plt.title('France VS Germany Cereal Production Bar plot')
    plt.xticks(x + bar_width * (len(countries) - 1) / 2, x,
               rotation=45)  # Set x-axis ticks every 5 years
    # Place legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('barplot.png', dpi=300)  # Save plot as image
    plt.show()


def boxplot(df, countries):
    """
    Plot a box plot of cereal production for specified countries.

    Parameters:
    df (DataFrame): DataFrame containing cereal production data.
    countries (list): List of countries to plot.

    Returns:
    None
    """
    plt.figure(figsize=(7, 4))

    # Prepare data for boxplot
    data = [df[country][::5] for country in countries]

    # Define colors for the boxes
    colors = ['blue', 'green', 'red', 'orange']

    # Create boxplot with colors
    bp = plt.boxplot(data, labels=countries, patch_artist=True)

    # Set colors for each box
    for box, color in zip(bp['boxes'], colors):
        box.set(facecolor=color)

    # Add labels and title
    plt.xlabel('Countries')
    plt.ylabel('kg per Hectare')
    plt.title('Box plot for Cereal Production Across Countries')

    # Add grid and show plot
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('boxplot.png', dpi=300)  # Save plot as image
    plt.show()


# Main Function
file_name = 'API_19_DS2_en_csv_v2_764.csv'
indi_name = 'Cereal yield (kg per hectare)'
countries_name = ['India', 'France', 'Germany', 'Japan']
processed_data, processed_data_trans = read_process_data(
    file_name, indi_name, countries_name)

lineplot(processed_data_trans, ['India', 'France', 'Germany', 'Japan'])
barplot(processed_data_trans, ['France', 'Germany'])
boxplot(processed_data_trans, ['India', 'France', 'Germany', 'Japan'])
