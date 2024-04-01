# NYC Real Estate Sales Dashboard

The NYC Real Estate Sales Dashboard is a powerful, interactive tool designed for real estate professionals, investors, and homebuyers seeking in-depth insights into the New York City real estate market. Leveraging data from the NYC Department of Finance's [Rolling Sales Data](https://www.nyc.gov/site/finance/taxes/property-rolling-sales-data.page), this application provides comprehensive analyses of real estate transactions across the five boroughs of New York City.

## Live Demo

Check out the live deployment of the NYC Real Estate Sales Dashboard on Render:

[https://nyc-re-dashboard.onrender.com/](https://nyc-re-dashboard.onrender.com/)

## Features

- **Exploratory Data Analysis (EDA)**: Dive into an interactive exploration of the data through various visualizations, including histograms, scatterplots, and time series analyses. Filter and compare sales activity and pricing across neighborhoods or ZIP codes.

- **Geospatial Analysis**: Visualize the geographical distribution of real estate sales and pricing patterns on an interactive map. Gain insights into spatial trends and identify hotspots across the city.

- **Comparative Analysis**: Compare key real estate metrics, such as average sale prices, total sales volume, and the number of sales, across the five boroughs. Adjust the visualizations to display per capita values for a more nuanced comparison.

- **Data Overview**: Explore summary statistics, frequency counts, and descriptive information about the real estate sales data for each borough. The application also provides explanations for building class codes, sourced from the [NYC Department of Finance](https://www.nyc.gov/assets/finance/jump/hlpbldgcode.html) and stored in a custom CSV file.

## Methods and Libraries

The NYC Real Estate Sales Dashboard is implemented using Python and leverages the following libraries:

- **Streamlit**: A powerful framework for building interactive web applications using Python. Streamlit allows for the creation of intuitive user interfaces and seamless integration of data visualizations.

- **pandas**: A fast, flexible, and expressive data manipulation library for Python. pandas is used for data loading, cleaning, transformation, and aggregation.

- **NumPy**: A fundamental package for scientific computing in Python. NumPy provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions.

- **Matplotlib**: A comprehensive library for creating static, animated, and interactive visualizations in Python. Matplotlib is used for creating various plots and charts in the application.

- **Seaborn**: A statistical data visualization library based on Matplotlib. Seaborn provides a high-level interface for drawing attractive and informative statistical graphics.

- **Folium**: A Python library for creating interactive maps using the Leaflet.js library. Folium is used for geospatial visualization and analysis in the application.

- **Requests**: A simple, yet elegant HTTP library for Python. Requests is used for making HTTP requests to retrieve data from the NYC Department of Finance's website.

- **openpyxl**: A Python library for reading and writing Excel files (xlsx/xlsm/xltx/xltm). openpyxl is used for loading the real estate sales data from Excel files.

## Data Sources

The NYC Real Estate Sales Dashboard relies on the following data sources:

- [NYC Department of Finance Rolling Sales Data](https://www.nyc.gov/site/finance/taxes/property-rolling-sales-data.page): This dataset contains information on real estate transactions in New York City, including sale prices, property characteristics, and geographical information.

- [NYC ZIP Code Tabulation Areas (ZCTAs) Polygons](https://data.cityofnewyork.us/Business/Zip-Code-Boundaries/i8iw-xf4u): This dataset provides geometric polygons representing the boundaries of ZIP Code Tabulation Areas (ZCTAs) in New York City. It is used for geospatial visualization and analysis.

- [US ZIP Code Geodata](https://github.com/OpenDataDE/State-zip-code-GeoJSON): This dataset contains latitude and longitude coordinates for US ZIP codes. It is used for plotting markers on the interactive map.

- [NYC Department of Finance Building Class Codes](https://www.nyc.gov/assets/finance/jump/hlpbldgcode.html): This webpage provides explanations for building class codes used in the real estate sales data. The information is extracted and stored in a custom CSV file (`buildingclasscode.csv`) for easy reference within the application.

## Getting Started

To launch the NYC Real Estate Sales Dashboard on your local machine, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/nyc-real-estate-sales-dashboard.git
   ```

2. Navigate to the project directory:
   ```
   cd nyc-real-estate-sales-dashboard
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   streamlit run app.py
   ```

5. Open your web browser and visit `http://localhost:10000` (streamlit should launch a browser window for you) to access the NYC Real Estate Sales Dashboard.

## Deploying on Render

To deploy the NYC Real Estate Sales Dashboard on Render, follow these steps:

1. Sign up for a free account on [Render](https://render.com/).

2. Connect your GitHub repository containing the project to Render.

3. Create a new web service on Render and select the repository.

4. Configure the following settings for the web service:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py`

5. Add the following environment variables in the Render web service settings:
   - `PYTHON_VERSION`: 3.9.0

6. Click on "Create Web Service" to deploy the application.

7. Once the deployment is complete, you will receive a unique URL for your deployed app.

Your NYC Real Estate Sales Dashboard is now live and accessible via the provided URL.

## Contributing

Contributions to the NYC Real Estate Sales Dashboard are welcome! If you have any ideas, suggestions, or bug reports, please open an issue on the [GitHub repository](https://github.com/your-username/nyc-real-estate-sales-dashboard/issues). If you'd like to contribute code, please fork the repository and submit a pull request with your changes.

## License

The NYC Real Estate Sales Dashboard is open-source software licensed under the [MIT license](https://opensource.org/licenses/MIT).

## Contact

For any inquiries or feedback, please contact the developer, (ME!) Rasmus Foyer, at rasmus.foyer@gmail.com or call at 917-753-5574.

Happy exploring and analyzing the NYC real estate market!