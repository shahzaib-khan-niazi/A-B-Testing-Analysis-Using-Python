# A-B-Testing-Analysis-Using-Python
This project analyzes an A/B test experiment comparing click-through rates between a control group and an experiment group. It demonstrates a full workflow from data cleaning and exploratory analysis to statistical comparison and logistic regression modeling.

📂 Dataset
File: ab_test_click_data.csv

Description: Contains user click data with group labels (control or experiment) and a timestamp (not needed for this analysis).

🔑 Objectives
Clean and prepare the dataset
Explore click distribution for each group
Visualize click counts and rates
Calculate key metrics: mean, variance, lift, odds ratio
Build and evaluate a logistic regression model to predict click likelihood based on group membership

🗂️ Key Steps
1)Data Cleaning
Removed unnecessary columns (timestamp)
Checked for duplicates and missing values
2)Exploratory Data Analysis
Calculated group-wise counts, means, variances, standard deviations
Visualized click distribution with bar charts, histograms, and a pie chart
3)Modeling
Encoded group as binary (0 = control, 1 = experiment)
Trained a logistic regression model to predict the probability of a click
Evaluated accuracy, precision, recall, and F1 score
4)Statistical Insights
Computed the lift in click-through rate for the experiment vs. control
Calculated the odds ratio to compare groups

📊 Results Highlights
The experiment group shows a measurable lift in click-through rate compared to the control group.
Logistic regression model provides probability estimates for user clicks based on group assignment.
Visualizations help understand overall click distribution and model predictions.
