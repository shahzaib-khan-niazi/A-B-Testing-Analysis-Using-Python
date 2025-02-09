# A-B-Testing-Analysis-Using-Python
A-B-Testing-Analysis-Using-Python
This Python script analyzes A/B testing data to determine whether a new version of a feature (or webpage) performs better than the old version in terms of user clicks. The script loads the dataset, processes it, and visualizes the results to compare the performance of the control group ("con") and the experimental group ("exp").

Key Features & Analysis Performed: ✅ 1. Data Loading & Initial Inspection:

Reads the A/B test data from ab_test_click_data (1).csv. Displays the first few rows (.head()) and summary statistics (.describe()). ✅ 2. Click Count Analysis for Each Group:

Counts the total number of users in each group (groupby("group")["click"].count()). Sums the total number of clicks per group (groupby("group")["click"].sum()). ✅ 3. Visualizing A/B Test Results:

Uses seaborn to create a bar chart comparing clicks for the control (con) and experimental (exp) groups. Applies a custom color palette: Yellow → No Click Black → Click ✅ 4. Annotating the Bar Chart with Click Percentages:

Calculates the percentage of users who clicked in each group. Displays the percentages on the bar chart for better interpretation. ✅ 5. Extracting and Printing Click Data:

Retrieves the exact number of clicks for the control and experimental groups. Prints the results to compare engagement levels. Python Libraries Used: 📌 pandas → Data processing & aggregation 📌 matplotlib.pyplot → Data visualization (bar charts) 📌 seaborn → Advanced statistical plotting 📌 numpy → Numerical operations

Possible Improvements & Fixes: 🚀 Statistical Significance Testing: Use Chi-square test or T-test to confirm if the difference is significant. 🚀 More Visualizations: Use pie charts or histograms to show user engagement. 🚀 Data Cleaning: Handle missing or incorrect values before analysis.
