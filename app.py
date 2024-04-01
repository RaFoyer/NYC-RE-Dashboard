# Standard library imports
from datetime import datetime
import calendar
import json
from io import BytesIO

# Third-party library imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import openpyxl
import folium
from folium.plugins import MarkerCluster
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from pandas.tseries.offsets import MonthEnd

# Streamlit specific imports
import streamlit as st
from streamlit_folium import folium_static

# Set the page configuration
st.set_page_config(layout="wide")

# Set the aesthetics for seaborn and matplotlib plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


@st.cache_data
def load_data(file_name):
    """
    Attempts to load data from a URL. If unsuccessful, tries to load from a local file.
    Wrapped with Streamlit's cache decorator to avoid reloading data on every interaction.
    """
    base_url = "https://www.nyc.gov/assets/finance/downloads/pdf/rolling_sales/"
    full_url = base_url + file_name
    
    try:
        response = requests.get(full_url)
        response.raise_for_status()
        data = pd.read_excel(BytesIO(response.content), engine='openpyxl', header=4)
    except requests.exceptions.RequestException:
        try:
            # Assumes the file is in the 'data' directory within the project folder
            local_path = 'data/' + file_name
            data = pd.read_excel(local_path, engine='openpyxl', header=4)
        except Exception as e:
            st.error(f"Failed to load data from both URL and local file. Error: {e}")
            data = None
    return data

@st.cache_data
def load_building_class_codes():
    """
    Loads the building class codes from the buildingclasscode.csv file.
    """
    try:
        building_class_codes = pd.read_csv("buildingclasscode.csv")
        return building_class_codes
    except Exception as e:
        st.error(f"Failed to load building class codes from CSV file. Error: {e}")
        return None

@st.cache_data
def prepare_time_series(df, date_column):
    """
    Prepares and adjusts the time series data for monthly sales volume.
    Converts 'sale_month' to the first day of the month in datetime format to ensure compatibility.
    """
    # Convert 'sale_date' to the first day of its month as datetime
    df['sale_month'] = pd.to_datetime(df[date_column]).dt.to_period('M').dt.to_timestamp()

    # Group by 'sale_month' to calculate monthly sales, ensuring 'sale_month' is datetime
    monthly_sales = df.groupby('sale_month').size()

    # Prepare the adjusted sales count considering the last month may be incomplete
    last_month = monthly_sales.index[-1].to_period('M')
    if df[date_column].max() < last_month.to_timestamp(how='end'):
        days_in_month = calendar.monthrange(last_month.year, last_month.month)[1]
        last_day_of_month = pd.Timestamp(year=last_month.year, month=last_month.month, day=days_in_month)
        days_passed = (df[date_column].max() - last_day_of_month).days
        prorate_factor = days_in_month / (days_in_month - days_passed)
        monthly_sales.iloc[-1] *= prorate_factor

    # Merge the monthly sales back into the original dataframe
    df = df.merge(monthly_sales.reset_index(name='adjusted_sales_count'), on='sale_month', how='left')
    return df



# Additional utility functions for data cleaning

@st.cache_data
def clean_column_names(df):
    """
    Standardizes column names: converts to lowercase and replaces spaces with underscores.
    """
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    return df

def remove_unnecessary_columns(df, columns_to_remove):
    """
    Removes specified columns from the DataFrame.
    """
    return df.drop(columns=columns_to_remove, errors='ignore')

@st.cache_data
def fill_missing_values(df, columns_to_fill, method='mean'):
    """
    Fills missing values in specified columns using the specified method (mean or median).
    """
    for col in columns_to_fill:
        if method == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        elif method == 'median':
            df[col] = df[col].fillna(df[col].median())
    return df

@st.cache_data
def convert_data_types(df, conversions):
    """
    Converts the data types of specified columns in the DataFrame.
    """
    for col, dtype in conversions.items():
        if col == 'zip_code' and dtype == 'str':
            # Special handling for zip_code to remove trailing decimals
            df[col] = df[col].apply(lambda x: str(int(x)) if pd.notnull(x) else None)
        else:
            df[col] = df[col].astype(dtype)
    return df


@st.cache_data
def filter_non_market_sales(df, threshold=10000):
    """
    Filters out transactions with a sale price below a certain threshold,
    considering them non-market sales. The default threshold is set to $10,000.
    
    Parameters:
    - df: The DataFrame containing real estate transactions.
    - threshold: The sale price threshold below which transactions are considered non-market.
    
    Returns:
    - A DataFrame excluding non-market sales transactions.
    """
    return df[df['sale_price'] >= threshold]

@st.cache_data
def drop_rows_without_zipcode(df):
    """
    Drops rows where the zip code is missing.
    
    Parameters:
    - df: The DataFrame from which to drop rows.
    
    Returns:
    - A DataFrame with rows missing zip code removed.
    """
    return df.dropna(subset=['zip_code'])

@st.cache_data
def add_price_per_sqft(df):
    """
    Adds a new column 'price_per_sqft' to the DataFrame, which represents the sale price per square foot.

    Parameters:
    - df: The DataFrame containing real estate transactions.

    Returns:
    - The modified DataFrame with the 'price_per_sqft' column added.
    """
    df['price_per_sqft'] = df['sale_price_millions'] * 1e6 / df['gross_square_feet']
    return df

def data_cleaning_pipeline(borough_dataframes):
    """
    Applies a series of data cleaning operations to each borough's DataFrame.
    """
    cleaning_message = st.empty()  # Placeholder for cleaning messages
    columns_to_remove = ['block', 'lot', 'easement', 'address', 'apartment_number',
                         'tax_class_at_present', 'tax_class_at_time_of_sale']
    columns_to_fill = ['residential_units', 'commercial_units', 'total_units',
                       'land_square_feet', 'gross_square_feet']
    data_type_conversions = {'zip_code': 'str'}
    
    for borough, df in borough_dataframes.items():
        cleaning_message.write(f"Cleaning data for {borough}...")
        df = clean_column_names(df)
        df = remove_unnecessary_columns(df, columns_to_remove)
        df = filter_non_market_sales(df)  # Filtering non-market sales
        df = fill_missing_values(df, columns_to_fill)
        df = drop_rows_without_zipcode(df)  # Drop rows without a zipcode
        df = convert_data_types(df, data_type_conversions)
        df['sale_price_millions'] = df['sale_price'] / 1e6
        df = add_price_per_sqft(df)
        
        # Prepare time series data
        df = prepare_time_series(df, 'sale_date')
        
        borough_dataframes[borough] = df
    
    cleaning_message.success("Data cleaning completed.")
    time.sleep(3)  # Delay for 3 seconds before clearing the message
    cleaning_message.empty()  # Clear the message

    return borough_dataframes


def display_home(borough_dataframes):
    # Clear Value Proposition
    st.title("NYC Real Estate Sales Dashboard")
    st.write("Welcome to the NYC Real Estate Sales Dashboard, a powerful tool designed for real estate professionals, investors, and homebuyers seeking in-depth insights into the New York City real estate market.")

    # Project Introduction
    st.header("Project Overview")
    st.write("The NYC Real Estate Sales Dashboard leverages data from the NYC Department of Finance's [Rolling Sales Data](https://www.nyc.gov/site/finance/taxes/property-rolling-sales-data.page) to provide comprehensive analyses of real estate transactions across the five boroughs of New York City. The data is loaded live from the official source, with a local backup from March 2024 used as a fallback, ensuring access to the most recent information available.")

    # Data Visualization Preview
    st.subheader("Get a Glimpse of the Insights")
    st.write("Here's a example of visualizations and analyses you can explore within the app:")

    # Create a custom dashboard
    dashboard = st.container()
    with dashboard:
        col1, col2, col3 = st.columns(3)

        # Average Sale Price by Borough
        with col1:
            avg_sale_price = {borough: df['sale_price'].mean() / 1e6 for borough, df in borough_dataframes.items()}
            avg_price_df = pd.DataFrame.from_dict(avg_sale_price, orient='index', columns=['Average Sale Price (Millions $)'])
            avg_price_df = avg_price_df.sort_values(by='Average Sale Price (Millions $)', ascending=False)

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.barplot(x='Average Sale Price (Millions $)', y=avg_price_df.index, data=avg_price_df, palette='viridis')
            ax.set_title('Average Sale Price by Borough', fontweight='bold')
            st.pyplot(fig)

        # Total Sales Volume by Borough
        with col2:
            total_sales_volume = {borough: df['sale_price'].sum() / 1e9 for borough, df in borough_dataframes.items()}
            sales_volume_df = pd.DataFrame.from_dict(total_sales_volume, orient='index', columns=['Total Sales Volume (Billions $)'])
            sales_volume_df = sales_volume_df.sort_values(by='Total Sales Volume (Billions $)', ascending=False)

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.barplot(x='Total Sales Volume (Billions $)', y=sales_volume_df.index, data=sales_volume_df, palette='viridis')
            ax.set_title('Total Sales Volume by Borough', fontweight='bold')
            st.pyplot(fig)

        # Number of Sales by Borough
        with col3:
            sales_count = {borough: len(df) for borough, df in borough_dataframes.items()}
            sales_count_df = pd.DataFrame.from_dict(sales_count, orient='index', columns=['Number of Sales'])
            sales_count_df = sales_count_df.sort_values(by='Number of Sales', ascending=False)

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.barplot(x='Number of Sales', y=sales_count_df.index, data=sales_count_df, palette='viridis')
            ax.set_title('Number of Sales by Borough', fontweight='bold')
            st.pyplot(fig)

    # Navigation Guide
    st.header("Navigation Guide")
    st.write("Use the sidebar to navigate through the different sections of the app. Each page offers unique visualizations and analyses tailored to specific aspects of the real estate market.")

    # Page Descriptions
    with st.expander("Page Descriptions"):
        st.subheader("Exploratory Data Analysis (EDA)")
        st.write("Dive into an interactive exploration of the data through various visualizations, including histograms, scatterplots, and time series analyses. Filter and compare sales activity and pricing across neighborhoods or ZIP codes.")

        st.subheader("Geospatial Analysis")
        st.write("Visualize the geographical distribution of real estate sales and pricing patterns on an interactive map. Gain insights into spatial trends and identify hotspots across the city.")

        st.subheader("Comparative Analysis")
        st.write("Compare key real estate metrics, such as average sale prices, total sales volume, and the number of sales, across the five boroughs. Adjust the visualizations to display per capita values for a more nuanced comparison.")

        st.subheader("Data Overview")
        st.write("Explore summary statistics, frequency counts, and descriptive information about the real estate sales data for each borough.")

    # Call to Action
    st.subheader("Get Started")
    st.write("Ready to unlock the power of data-driven real estate insights? Click the button below to begin your exploration!")
    start_button = st.button("Explore the Data")

    # Navigate to EDA page if the button is clicked
    if start_button:
        display_eda(borough_dataframes)

    # User Feedback and Engagement
    st.subheader("Feedback and Support")
    st.write("I value your feedback and suggestions. If you have any questions or would like to share your experience with the app, please email me at rasmus.foyer@gmail.com or call me at 917-753-5574.")


    # Example of displaying a DataFrame in Streamlit
    example_borough = st.selectbox("Select a borough to display sample data:", list(borough_dataframes.keys()))
    st.write(borough_dataframes[example_borough].sample(5))


def display_data_overview(borough_dataframes):
    st.title("Data Overview")
    st.write("This section provides an overview of the NYC Real Estate Sales data for each borough, including descriptive statistics, frequency counts, and an explanation of building codes.")

    borough_selection = st.selectbox("Select a borough to view its data overview:", list(borough_dataframes.keys()))
    df = borough_dataframes[borough_selection]

    # Handle numerical and categorical columns separately
    numerical_cols = ['sale_price', 'land_square_feet', 'gross_square_feet', 'total_units']
    categorical_cols = ['zip_code', 'building_class_at_present']

    st.write("### Descriptive Statistics for Numerical Data")
    st.dataframe(df[numerical_cols].describe().transpose())


    st.write("### Frequency Counts for Categorical Data")

    cols = st.columns(len(categorical_cols))
    displayed_codes = []  # Initialize an empty list to keep track of displayed building class codes
    for idx, col in enumerate(categorical_cols):
        with cols[idx]:
            st.markdown(f"**{col.replace('_', ' ').title()}**")
            if col == 'building_class_at_present':
                num_codes_to_display = st.slider("Number of most frequent building class codes to display:", min_value=1, max_value=20, value=6)
                code_counts = df[col].value_counts().reset_index().rename(columns={'index': 'Code', col: 'Count'})
                displayed_df = code_counts.head(num_codes_to_display)
                st.dataframe(displayed_df)
                displayed_codes = displayed_df['Count'].tolist()

            else:
                st.dataframe(df[col].value_counts().reset_index().rename(columns={'index': col, col: 'Count'}).head(20))

    st.write("### Building Class Codes Explanation")

    # Load the building class codes from the CSV file
    building_class_codes = load_building_class_codes()
    if building_class_codes is not None and displayed_codes:
        # Filter the building class codes to only include the displayed codes
        present_codes_df = building_class_codes[building_class_codes['id'].isin(displayed_codes)]
        
        if not present_codes_df.empty:
            st.write("Here's what the displayed building codes mean:")
            st.table(present_codes_df.set_index('id')['name'])
        else:
            st.write("No building class codes present in the displayed data.")
    else:
        st.write("Failed to load building class codes or no codes to display. Please check the data file.")

    st.write("Source: [NYC Department of Finance](https://www.nyc.gov/assets/finance/jump/hlpbldgcode.html)")


def display_eda(borough_dataframes):
    st.title("Exploratory Data Analysis (EDA)")
    st.write("Explore various insights derived from the NYC Real Estate Sales data.")

    # Allow the user to select a borough for EDA
    borough_selection = st.selectbox("Select a borough for EDA:", list(borough_dataframes.keys()), key='eda_borough')
    df = borough_dataframes[borough_selection].copy()

    df['price_per_sqft'] = df['sale_price_millions'] * 1e6 / df['gross_square_feet']

    # User input for adjusting histogram bins and outlier cutoff
    bins = st.slider("Select the number of bins for the histogram:", min_value=10, max_value=100, value=50, step=5, key='hist_bins')
    enable_cutoff = st.checkbox("Enable outlier cutoff adjustment", value=True)
    cutoff_multiplier = 3.0  # Default value
    if enable_cutoff:
        cutoff_multiplier = st.slider("Define the outlier cutoff multiplier (standard deviations from the mean):", min_value=1.0, max_value=5.0, value=3.0, step=0.5, key='outlier_cutoff')

    # Apply cutoff multiplier before further analyses
    mean_val = df['sale_price_millions'].mean()
    std_val = df['sale_price_millions'].std()
    low_bound = mean_val - (cutoff_multiplier * std_val)
    high_bound = mean_val + (cutoff_multiplier * std_val)
    filtered_df = df[(df['sale_price_millions'] >= low_bound) & (df['sale_price_millions'] <= high_bound)]

    # Calculate mean and median of the filtered dataset for the vertical lines
    mean_val = filtered_df['sale_price_millions'].mean()
    median_val = filtered_df['sale_price_millions'].median()

    # Plotting the histogram with user-defined parameters
    fig, ax = plt.subplots()
    sns.histplot(filtered_df['sale_price_millions'], bins=bins, kde=True, ax=ax)
    ax.set_title('Distribution of Sale Prices' + (' (with Outlier Cutoff)' if enable_cutoff else ''))
    ax.set_xlabel('Sale Price (Millions $)')
    ax.set_ylabel('Frequency')

    # Add vertical lines for mean and median now that they're defined
    ax.axvline(mean_val, color='r', linestyle='--', label=f'Mean: ${mean_val:.2f}M')
    ax.axvline(median_val, color='g', linestyle='-', label=f'Median: ${median_val:.2f}M')
    ax.legend()

    st.pyplot(fig)


    # Additional filtering options for neighborhood or zip code with "All" option
    filter_type = st.radio("Further filter by:", ["Neighborhood", "Zip Code"], key='filter_type')
    options = ["All Neighborhoods"] if filter_type == "Neighborhood" else ["All Zip Codes"]
    options += sorted(df[filter_type.lower().replace(' ', '_')].unique())

    selected_options = st.multiselect(f"Select {filter_type.lower()}(s):", options, key=f"{filter_type.lower()}_select", default=options[0])

    # Splitting the layout into two columns for cumulative and comparative analysis
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"#### Cumulative Analysis")
        
        # Prepare data frame for cumulative analysis
        if "All Neighborhoods" in selected_options or "All Zip Codes" in selected_options or not selected_options:
            cumulative_df = filtered_df
        else:
            filter_column = 'neighborhood' if filter_type == "Neighborhood" else 'zip_code'
            cumulative_df = filtered_df[filtered_df[filter_column].isin(selected_options)]
        
        # Plot cumulative sale price distribution
        if not cumulative_df.empty:
            bins_cumulative = st.slider("Number of bins (Cumulative Histogram):", min_value=10, max_value=100, value=30, step=5)
            fig, ax = plt.subplots()
            sns.histplot(cumulative_df['sale_price_millions'], bins=bins_cumulative, kde=True, ax=ax)
            ax.set_title('Cumulative Sale Price Distribution', fontsize=16, fontweight='bold')
            ax.set_xlabel('Sale Price (Millions $)')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
        else:
            st.write("No data available for selected options.")
        
        # Plot cumulative sales volume over time
        if not cumulative_df.empty:
            cumulative_sales = cumulative_df.groupby('sale_month').size().reset_index(name='sales_count')
            cumulative_sales['sale_month'] = pd.to_datetime(cumulative_sales['sale_month'].astype(str))
            fig, ax = plt.subplots()
            sns.lineplot(data=cumulative_sales, x='sale_month', y='sales_count', marker='o', ax=ax)
            ax.set_title('Cumulative Sales Volume Over Time', fontsize=16, fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel('Sales Count')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.write("No data available for selected options.")
        
        # Plot cumulative average sales price over time
        if not cumulative_df.empty:
            cumulative_avg_price = cumulative_df.groupby('sale_month')['sale_price_millions'].mean().reset_index()
            cumulative_avg_price['sale_month'] = pd.to_datetime(cumulative_avg_price['sale_month'].astype(str))
            fig, ax = plt.subplots()
            sns.lineplot(data=cumulative_avg_price, x='sale_month', y='sale_price_millions', marker='o', ax=ax)
            ax.set_title('Cumulative Average Sales Price Over Time', fontsize=16, fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel('Average Sales Price (Millions $)')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.write("No data available for selected options.")
        
        # Plot cumulative sale price boxplot
        if not cumulative_df.empty:
            fig, ax = plt.subplots()
            sns.boxplot(x=cumulative_df['sale_price_millions'], ax=ax)
            ax.set_title('Cumulative Sale Price Boxplot', fontsize=16, fontweight='bold')
            ax.set_xlabel('Sale Price (Millions $)')
            st.pyplot(fig)
        else:
            st.write("No data available for selected options.")
        
        # Add a checkbox to filter out data points with imputed gross square feet
        filter_imputed_cumulative = st.checkbox("Filter out data points with imputed gross square feet (Cumulative)", value=True)

        if filter_imputed_cumulative:
            # Identify the most frequent values in the 'gross_square_feet' column
            frequent_values = cumulative_df['gross_square_feet'].value_counts().head(5).index.tolist()
            
            # Filter out data points where 'gross_square_feet' is in the list of frequent values
            cumulative_df = cumulative_df[~cumulative_df['gross_square_feet'].isin(frequent_values)]

        # Plot cumulative sale price vs. gross square feet scatterplot
        if not cumulative_df.empty:
            fig, ax = plt.subplots()
            sns.scatterplot(data=cumulative_df, x='gross_square_feet', y='sale_price_millions', ax=ax)
            ax.set_title('Cumulative Sale Price vs. Gross Square Feet', fontsize=16, fontweight='bold')
            ax.set_xlabel('Gross Square Feet')
            ax.set_ylabel('Sale Price (Millions $)')
            st.pyplot(fig)
        else:
            st.write("No data available for selected options.")

                # Plot cumulative sale price per square foot scatterplot
        if not cumulative_df.empty:
            filter_imputed_cumulative_ppsf = st.checkbox("Filter out data points with imputed gross square feet (Cumulative Price/SF)", value=True)

            if filter_imputed_cumulative_ppsf:
                # Identify the most frequent values in the 'gross_square_feet' column
                frequent_values = cumulative_df['gross_square_feet'].value_counts().head(5).index.tolist()

                # Filter out data points where 'gross_square_feet' is in the list of frequent values
                cumulative_df = cumulative_df[~cumulative_df['gross_square_feet'].isin(frequent_values)]

            if not cumulative_df.empty:
                fig, ax = plt.subplots()
                sns.scatterplot(data=cumulative_df, x='gross_square_feet', y='price_per_sqft', ax=ax)
                ax.set_title('Cumulative Sale Price per Square Foot', fontsize=16, fontweight='bold')
                ax.set_xlabel('Gross Square Feet')
                ax.set_ylabel('Sale Price per Square Foot ($)')
                st.pyplot(fig)
            else:
                st.write("No data available for selected options.")

    with col2:
        st.write(f"#### Comparative Analysis")
        
        filter_column = 'neighborhood' if filter_type == "Neighborhood" else 'zip_code'
        
        # Prepare data frame for comparative analysis
        if "All Neighborhoods" in selected_options or "All Zip Codes" in selected_options or not selected_options:
            comparative_df = filtered_df
        else:
            comparative_df = filtered_df[filtered_df[filter_column].isin(selected_options)]
        
        # Plot comparative sale price distribution
        if not comparative_df.empty:
            bins_comparative = st.slider("Number of bins (Comparative Histogram):", min_value=10, max_value=100, value=30, step=5)
            fig, ax = plt.subplots()
            for option in selected_options:
                if option in ["All Neighborhoods", "All Zip Codes"]:
                    continue  # Skip plotting for "All" options
                option_df = comparative_df[comparative_df[filter_column] == option]
                sns.histplot(option_df['sale_price_millions'], bins=bins_comparative, kde=True, ax=ax, label=option, alpha=0.6)
            ax.set_title('Comparative Sale Price Distribution', fontsize=16, fontweight='bold')
            ax.set_xlabel('Sale Price (Millions $)')
            ax.set_ylabel('Frequency')
            ax.legend(title=filter_type)
            st.pyplot(fig)
        else:
            st.write("No data available for selected options.")
        
        # Plot comparative sales volume over time
        if not comparative_df.empty:
            comparative_sales = comparative_df.groupby(['sale_month', filter_column]).size().reset_index(name='sales_count')
            comparative_sales['sale_month'] = pd.to_datetime(comparative_sales['sale_month'].astype(str))
            fig, ax = plt.subplots()
            for option in selected_options:
                if option in ["All Neighborhoods", "All Zip Codes"]:
                    continue  # Skip plotting for "All" options
                option_df = comparative_sales[comparative_sales[filter_column] == option]
                sns.lineplot(data=option_df, x='sale_month', y='sales_count', marker='o', label=option, ax=ax)
            ax.set_title('Comparative Sales Volume Over Time', fontsize=16, fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel('Sales Count')
            ax.legend(title=filter_type)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.write("No data available for selected options.")
        
        # Plot comparative average sales price over time
        if not comparative_df.empty:
            comparative_avg_price = comparative_df.groupby(['sale_month', filter_column])['sale_price_millions'].mean().reset_index()
            comparative_avg_price['sale_month'] = pd.to_datetime(comparative_avg_price['sale_month'].astype(str))
            fig, ax = plt.subplots()
            for option in selected_options:
                if option in ["All Neighborhoods", "All Zip Codes"]:
                    continue  # Skip plotting for "All" options
                option_df = comparative_avg_price[comparative_avg_price[filter_column] == option]
                sns.lineplot(data=option_df, x='sale_month', y='sale_price_millions', marker='o', label=option, ax=ax)
            ax.set_title('Comparative Average Sales Price Over Time', fontsize=16, fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel('Average Sales Price (Millions $)')
            ax.legend(title=filter_type)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.write("No data available for selected options.")
        
        # Plot comparative sale price boxplot
        if not comparative_df.empty:
            fig, ax = plt.subplots()
            sns.boxplot(x=filter_column, y='sale_price_millions', data=comparative_df, ax=ax)
            ax.set_title('Comparative Sale Price Boxplot', fontsize=16, fontweight='bold')
            ax.set_xlabel(filter_type)
            ax.set_ylabel('Sale Price (Millions $)')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.write("No data available for selected options.")
        
        # Add a checkbox to filter out data points with imputed gross square feet
        filter_imputed_comparative = st.checkbox("Filter out data points with imputed gross square feet (Comparative)", value=True)

        if filter_imputed_comparative:
            # Identify the most frequent values in the 'gross_square_feet' column
            frequent_values = comparative_df['gross_square_feet'].value_counts().head(5).index.tolist()
            
            # Filter out data points where 'gross_square_feet' is in the list of frequent values
            comparative_df = comparative_df[~comparative_df['gross_square_feet'].isin(frequent_values)]

        # Plot comparative sale price vs. gross square feet scatterplot
        if not comparative_df.empty:
            fig, ax = plt.subplots()
            sns.scatterplot(data=comparative_df, x='gross_square_feet', y='sale_price_millions', hue=filter_column, ax=ax)
            ax.set_title('Comparative Sale Price vs. Gross Square Feet', fontsize=16, fontweight='bold')
            ax.set_xlabel('Gross Square Feet')
            ax.set_ylabel('Sale Price (Millions $)')
            st.pyplot(fig)
        else:
            st.write("No data available for selected options.")

            # Plot comparative sale price per square foot scatterplot
        if not comparative_df.empty:
            filter_imputed_comparative_ppsf = st.checkbox("Filter out data points with imputed gross square feet (Comparative Price/SF)", value=True)

            if filter_imputed_comparative_ppsf:
                # Identify the most frequent values in the 'gross_square_feet' column
                frequent_values = comparative_df['gross_square_feet'].value_counts().head(5).index.tolist()

                # Filter out data points where 'gross_square_feet' is in the list of frequent values
                comparative_df = comparative_df[~comparative_df['gross_square_feet'].isin(frequent_values)]

            if not comparative_df.empty:
                fig, ax = plt.subplots()
                sns.scatterplot(data=comparative_df, x='gross_square_feet', y='price_per_sqft', hue=filter_column, ax=ax)
                ax.set_title('Comparative Sale Price per Square Foot', fontsize=16, fontweight='bold')
                ax.set_xlabel('Gross Square Feet')
                ax.set_ylabel('Sale Price per Square Foot ($)')
                st.pyplot(fig)
            else:
                st.write("No data available for selected options.")

def display_geospatial_analysis(borough_dataframes):
    st.title("Geospatial Analysis")
    st.write("""
        This section offers a geospatial visualization of New York City's real estate market, focusing on sales activity and price distributions across different boroughs. By integrating ZIP code-level real estate data with geospatial information, users can examine:

        - **Average sale prices and total sales volume** by ZIP code, visualized through a color-coded map.
        - **Detailed metrics for specific ZIP codes**, including residential and commercial units, land and gross square footage, accessible via interactive markers and tooltips.
        - **Comparative insights** with borough-wide and city-wide sales data, facilitating market trend analysis and investment decision-making.

        Designed for real estate professionals, analysts, and investors, this tool aids in identifying spatial trends, market hotspots, and areas of potential interest based on comprehensive sales data. The interactive map enhances data exploration, offering a direct way to assess market dynamics and inform strategic planning.
        """)
    
    # Load ZIP code geospatial data
    geodata_df = pd.read_csv("NYC_ZIP_geodata/uszipcodes_geodata.txt", dtype={'ZIP': str})
    
    # Aggregate data from all boroughs
    all_data = pd.concat(borough_dataframes.values(), ignore_index=True)
    
    # Perform aggregation by ZIP code
    zipcodes_agg = all_data.groupby('zip_code').agg({
        'sale_price': ['mean', 'sum'],
        'residential_units': 'mean',
        'commercial_units': 'mean',
        'total_units': 'mean',
        'land_square_feet': 'mean',
        'gross_square_feet': 'mean',
        'borough': 'first',
        'zip_code': 'count'
    }).reset_index()
    
    # Simplify column names after aggregation
    zipcodes_agg.columns = ['zip_code', 'mean_sale_price', 'total_sale_volume', 'mean_residential_units',
                            'mean_commercial_units', 'mean_total_units', 'mean_land_square_feet',
                            'mean_gross_square_feet', 'borough', 'sales_count']
    
    # Calculate total sales volume and count for each borough and all of NYC
    borough_sales_volume = zipcodes_agg.groupby('borough')['total_sale_volume'].sum().reset_index()
    borough_sales_count = zipcodes_agg.groupby('borough')['sales_count'].sum().reset_index()
    nyc_total_sales_volume = zipcodes_agg['total_sale_volume'].sum()
    nyc_total_sales_count = zipcodes_agg['sales_count'].sum()
    
    # Merge the aggregated data with geospatial data
    zipcodes_agg = pd.merge(zipcodes_agg, geodata_df, left_on='zip_code', right_on='ZIP', how='left')
    
    # Load and prepare GeoJSON data for ZIP codes
    with open("NYC_ZIP_geodata/nyc-zip-code-tabulation-areas-polygons.geojson") as f:
        geo_data = json.load(f)
    
    # Initialize the map
    map_nyc = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
    
    # Calculate quantiles for mean sale price
    quantiles = zipcodes_agg['mean_sale_price'].quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1])
    
    # Overlay ZIP code boundaries with mean sale price coloring
    folium.Choropleth(
        geo_data=geo_data,
        name='choropleth',
        data=zipcodes_agg,
        columns=['zip_code', 'mean_sale_price'],
        key_on='feature.properties.postalCode',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Mean Sale Price ($)',
        bins=quantiles,
        max_zoom=11
    ).add_to(map_nyc)
    
    # Checkboxes to show or hide markers and tooltips
    show_markers = st.checkbox("Show Markers", value=True)
    show_tooltips = st.checkbox("Show Tooltips", value=True)

    # Conditional logic to add markers and tooltips based on checkbox values
    if show_markers:
        for idx, row in zipcodes_agg.iterrows():
            if pd.notnull(row['LAT']) and pd.notnull(row['LNG']):  # Ensure lat and lng are not null
                
                # Calculate borough and NYC totals for the current row
                borough_sales_volume_value = zipcodes_agg[zipcodes_agg['borough'] == row['borough']]['total_sale_volume'].sum()
                borough_sales_count_value = zipcodes_agg[zipcodes_agg['borough'] == row['borough']]['sales_count'].sum()

                # Pre-calculate percentages to avoid division by zero
                percent_borough_volume = (row['total_sale_volume'] / borough_sales_volume_value * 100) if borough_sales_volume_value else 0
                percent_nyc_volume = (row['total_sale_volume'] / nyc_total_sales_volume * 100) if nyc_total_sales_volume else 0
                percent_borough_count = (row['sales_count'] / borough_sales_count_value * 100) if borough_sales_count_value else 0
                percent_nyc_count = (row['sales_count'] / nyc_total_sales_count * 100) if nyc_total_sales_count else 0

                # Prepare tooltip content using pre-calculated percentages
                tooltip_content = (
                    f"<strong>ZIP Code:</strong> {row['ZIP']}<br>"
                    f"<strong>Mean Sale Price:</strong> ${int(row['mean_sale_price']):,}<br>"
                    f"<strong>Total Sales Volume:</strong> ${int(row['total_sale_volume']):,}<br>"
                    f"<strong>Sales Count:</strong> {int(row['sales_count'])}<br>"
                    f"<strong>% of Borough Sales Volume:</strong> {percent_borough_volume:.2f}%<br>"
                    f"<strong>% of NYC Sales Volume:</strong> {percent_nyc_volume:.2f}%<br>"
                    f"<strong>% of Borough Sales Count:</strong> {percent_borough_count:.2f}%<br>"
                    f"<strong>% of NYC Sales Count:</strong> {percent_nyc_count:.2f}%"
                )
                
                # Decide whether to use the tooltip based on the checkbox
                tooltip = folium.Tooltip(tooltip_content) if show_tooltips else None

                folium.Marker(
                    [row['LAT'], row['LNG']],
                    popup=folium.Popup(tooltip_content, max_width=300),
                    tooltip=tooltip
                ).add_to(map_nyc)

    # Display the map in Streamlit
    folium_static(map_nyc)



def display_comparative_analysis(borough_dataframes):
    st.title("Comparative Analysis")
    st.write("Compare real estate metrics across different boroughs to identify trends and outliers.")
    st.write("Source: https://www.citypopulation.de/en/usa/newyorkcity/")

    # Population data
    borough_populations = {
        "Bronx": 1356476,
        "Brooklyn": 2561225,
        "Manhattan": 1597451,
        "Queens": 2252196,
        "Staten Island": 490687
    }

    # Prepare data for comparison
    comparison_metrics = {
        'Average Sale Price (Millions $)': [],
        'Total Sales Volume (Millions $)': [],
        'Number of Sales': []
    }
    borough_names = list(borough_dataframes.keys())
    num_boroughs = len(borough_names)

    # Define a color map for the boroughs
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    available_colors = list(colors.values())

    # Extend the available colors if needed
    while len(available_colors) < num_boroughs:
        available_colors.extend(mcolors.random_rgb())

    borough_colors = {borough: available_colors[i] for i, borough in enumerate(borough_names)}

    for borough, df in borough_dataframes.items():
        comparison_metrics['Average Sale Price (Millions $)'].append(df['sale_price_millions'].mean())
        comparison_metrics['Total Sales Volume (Millions $)'].append(df['sale_price_millions'].sum())
        comparison_metrics['Number of Sales'].append(len(df))

    # Choose metric for comparison
    metric_to_compare = st.selectbox('Choose a metric to compare', list(comparison_metrics.keys()))

    # Checkbox to enable per capita comparison
    enable_per_capita = st.checkbox('Enable per capita comparison')

    # Generate and display the comparison bar chart
    fig, ax = plt.subplots()
    if enable_per_capita:
        bar_values = [comparison_metrics[metric_to_compare][i] / borough_populations[borough] for i, borough in enumerate(borough_names)]
    else:
        bar_values = comparison_metrics[metric_to_compare]

    ax.bar(borough_names, bar_values, color=[borough_colors[borough] for borough in borough_names])
    ax.set_ylabel(f'{metric_to_compare} {"(Per Capita)" if enable_per_capita else ""}')
    ax.set_title(f'{metric_to_compare} by Borough {"(Per Capita)" if enable_per_capita else ""}')
    ax.set_xticks(np.arange(len(borough_names)))
    ax.set_xticklabels(borough_names, rotation=45, ha='right')
    st.pyplot(fig)

    # Plot a pie chart for the distribution of sales across boroughs
    fig, ax = plt.subplots()
    sales_count = [df.shape[0] for df in borough_dataframes.values()]
    ax.pie(sales_count, labels=borough_names, autopct='%1.1f%%', colors=[borough_colors[borough] for borough in borough_names])
    ax.axis('equal')  # Ensure the pie chart is circular
    ax.set_title('Distribution of Sales Across Boroughs')
    st.pyplot(fig)

def display_about():
    st.title("About the Developer")
    st.write("""
    This application was developed by Rasmus Foyer, who brings over a decade of experience as an entrepreneur, in business, and as a real estate professional. Leveraging a rich background across various industries, Rasmus focuses on applying data science to solve complex challenges and create innovative solutions.

    For inquiries, feedback, or to discuss potential collaborations, you are welcome to get in touch:

    - **Email**: [rasmus.foyer@gmail.com](mailto:rasmus.foyer@gmail.com)
    - **Phone**: 917-753-5574
    - **GitHub**: [Visit Rasmus's GitHub](https://github.com/RasmusFoyer)

    Your feedback and inquiries are greatly appreciated.
    """)

def main():
    st.sidebar.title("Navigation")
    page_selection = st.sidebar.radio("Select a Page:", ["Home", "Data Overview", "EDA", "Geospatial Analysis", "Comparative Analysis", "About"])
    
    # Load the data for each borough
    boroughs = ["Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"]
    file_names = {
        "Manhattan": "rollingsales_manhattan.xlsx",
        "Bronx": "rollingsales_bronx.xlsx",
        "Brooklyn": "rollingsales_brooklyn.xlsx",
        "Queens": "rollingsales_queens.xlsx",
        "Staten Island": "rollingsales_statenisland.xlsx"
    }

    # Using Streamlit's caching mechanism to load and clean data only once
    @st.cache_data
    def load_and_clean_data():
        loading_message = st.sidebar.empty()  # Create a placeholder for loading messages
        borough_dataframes = {}
        for borough in boroughs:
            # Update the placeholder with the current loading status
            loading_message.write(f"Loading data for {borough}...")
            df = load_data(file_names[borough])
            if df is not None:
                # Optionally update the placeholder to indicate successful loading, or clear it
                loading_message.write(f"Data for {borough} loaded successfully.")
                borough_dataframes[borough] = df
            else:
                st.sidebar.error(f"Failed to load data for {borough}.")
                break  # Exit the loop if data loading fails
        loading_message.empty()  # Clear or replace the placeholder after loading is done
        return data_cleaning_pipeline(borough_dataframes)
    
    borough_dataframes = load_and_clean_data()
    
     # Page Routing
    if page_selection == "Home":
        display_home(borough_dataframes)
    elif page_selection == "Data Overview":
        display_data_overview(borough_dataframes)
    elif page_selection == "EDA":
        display_eda(borough_dataframes)
    elif page_selection == "Geospatial Analysis":
        display_geospatial_analysis(borough_dataframes)
    elif page_selection == "Comparative Analysis":
        display_comparative_analysis(borough_dataframes)
    elif page_selection == "About":
        display_about()


if __name__ == "__main__":
    main()