# The Anatomy of a Power Outage

The U.S. Department of Energy (DOE) provides critical information about the status and impacts of energy sector disruptions through the **Environment for Analysis of Geo-Located Energy Information (EAGLE-I)** system, operated by Oak Ridge National Laboratory. EAGLE-I supports monitoring of energy infrastructure assets, reporting of power outages, visualization of threats to energy infrastructure, and coordination of emergency response and recovery efforts.

Effectively responding to and restoring power during disasters depends on having timely, accurate, and actionable data.

In this exercise, your goal is to learn about K-Means and how clustering data can help you characterize it. This activity is based on one of the exercises from a larger data bootcamp that seeks to identify the characteristics and causes of power outages in the United States.

## Background 

K-Means is a machine learning algorithm used for clustering, which means grouping data points that are similar to each other. Instead of having predefined labels (as in classification), K-Means finds structure in the data on its own.
- You choose a number of clusters, K.
- The algorithm finds centers (called centroids) for those clusters.
- Each data point is assigned to the cluster with the nearest centroid.
- The centroids are updated until the clusters stabilize.

In this lab, we’ll use K-Means to group power outages based on their location (latitude and longitude) and time of year. This way, we can discover natural patterns, like whether outages cluster in certain regions or seasons.

## Why Normalize the Data?

Normalization (or scaling) is important because K-Means relies on distances to decide which points are similar. If one variable has a much larger numeric range than another, it can dominate the distance calculation.

For example:
- longitude values range roughly from -180 to 180.
- Time of year (say, months of the year 1–12) has a different scale.


If we don’t normalize, the clustering will be biased toward the feature with larger numbers.

By normalizing, we put all features on a comparable scale so that location and time both contribute fairly to the clustering.

## What is the best number of clusters to choose for a given set of data?


### The Elbow Method
- K-Means needs you to choose K, the number of clusters. But how do you know the best K?
- For each choice of K, we can calculate the within-cluster sum of squares (WCSS), which measures how close the points are to their cluster centers.
- As K increases, WCSS always goes down (more clusters = tighter groups).
- The trick is to look for the elbow in the WCSS vs. K graph:
  - At first, adding more clusters makes the WCSS drop a lot.
  - After a certain point, the improvement slows down.
  - That “elbow” point is often a good choice for K.

There are other tests too that you will find in the exercises in this notebook.

## How to Use This Notebook

You must activate the code cells in the notebook below for the code to be used. In some cases, you are only reading functions into the Python interpreter, and they will not produce output until you call the function in another cell.

To activate a cell, click on it, and then hold down "Shift" while also pressing "Enter" or "Return" on the keyboard.

## Tips for Using Jupyter Notebooks and Python

- If you get a pink error box after running a cell, scroll to the **bottom** of the error box to see what the main error is.  
- Often, if you get an error box, it’s because you skipped activating a cell above your current cell.  
- You can use an AI assistant to help you understand what the error message means—just make sure to **paste both the code that generated the error and the entire error message** into the AI for the best explanation.

