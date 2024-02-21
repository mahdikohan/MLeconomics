# Country Clustering Project

## Overview
This project aims to cluster countries based on various features to uncover patterns and similarities among them. By utilizing data analysis and machine learning techniques, we intend to group countries according to certain characteristics, facilitating insights into global trends and relationships.

## Features
1. **Log(GDP):** Taking the logarithm of GDP helps to normalize the distribution and better capture variations in economic output among countries.
2. **Oil Export Capacity:** The capacity for exporting oil is a significant economic factor for oil-producing countries, influencing their economic stability and global influence.
3. **GDP per Capita:** GDP per capita measures the economic output per person and provides insights into the average standard of living within a country.
4. **Regression Parameter of Log GDP:** The regression parameter of log GDP reflects the growth rate of a country's economy over time, offering insights into its economic development trajectory.

## Methodology
We utilized a correlation matrix to identify and remove additional features that may introduce multicollinearity issues or redundancy in the clustering analysis. Features with high correlations (above a predefined threshold) were excluded to ensure the independence of variables used for clustering.

![Correlation Matrix](mahdikohan/mlEconomics/countriesClustring/corr.png)

Next, we employed the K-means clustering algorithm to group countries based on the selected features. K-means is a centroid-based clustering algorithm that partitions data into K clusters, where each data point belongs to the cluster with the nearest mean. The process involves iteratively optimizing cluster centroids to minimize the within-cluster sum of squared distances.

## Dataset
The project utilizes a comprehensive dataset containing information on different countries and their corresponding features. The dataset encompasses data from reputable sources such as the World Bank, United Nations, and other international organizations.

## Usage
To replicate the clustering analysis:
1. Ensure Python and necessary libraries (e.g., Pandas, NumPy, Scikit-learn) are installed.
2. Clone this repository to your local machine.
3. Navigate to the project directory and run the provided Python script.
4. Follow the prompts to preprocess the data, choose clustering algorithms, and analyze the results.

## Results
Upon completion of the clustering analysis, the project will provide insights into distinct country clusters based on the selected features. Visualizations and summary statistics will be presented to facilitate interpretation and understanding of the findings.

## Future Work
Future iterations of this project may explore additional features, refine clustering algorithms, or incorporate more sophisticated machine learning techniques. Furthermore, comparisons with historical data and predictive modeling could offer valuable insights into the evolution of global socio-economic dynamics.