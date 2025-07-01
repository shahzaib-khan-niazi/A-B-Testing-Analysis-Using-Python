📈 A/B Testing Analysis Using Python
This project analyzes an A/B test comparing click-through rates between a control group and an experiment group. It demonstrates a full workflow from data cleaning and exploratory analysis to statistical comparison and logistic regression modeling.

📂 Dataset
File: ab_test_click_data.csv

Description: User click data with group labels (control, experiment) and a timestamp (dropped for this analysis).

🎯 Objectives
✅ Clean and prepare the dataset
✅ Explore click distribution for each group
✅ Visualize click counts and rates
✅ Calculate key metrics (mean, variance, lift, odds ratio)
✅ Build and evaluate a logistic regression model to predict clicks

🛠️ Key Steps
1️⃣ Data Cleaning
Removed unnecessary columns (timestamp)

Checked for duplicates and missing values

2️⃣ Exploratory Data Analysis
Calculated group-wise counts, means, variances

Visualized distributions with bar charts, histograms, and pie charts

3️⃣ Modeling
Encoded groups (0 = control, 1 = experiment)

Trained a logistic regression model to predict click probability

Evaluated accuracy, precision, recall, and F1 score

4️⃣ Statistical Insights
Computed lift in click-through rate (experiment vs. control)

Calculated odds ratio between groups

📊 Results Highlights
✅ The experiment group showed a measurable lift in click-through rates.
✅ Logistic regression provided click probability estimates using group assignment.
✅ Visualizations clarified user click distribution and model predictions.

🚀 Takeaway
This project demonstrates a practical A/B testing analysis workflow using Python, combining statistical methods and predictive modeling for clear, actionable insights.
