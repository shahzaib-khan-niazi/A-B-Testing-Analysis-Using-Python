import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.api as sm
ab_test_data = pd.read_csv("ab_test_click_data (1).csv")
print(ab_test_data.head())
print(ab_test_data.describe())
print(ab_test_data.groupby("group")["click"].count())
print(ab_test_data.groupby("group")["click"].sum())
palette = {0: 'lightgray', 1: 'black'}
ax = sns.countplot(x='group', hue='click', data=ab_test_data, palette=palette)
plt.xlabel("group")

plt.ylabel("count")
plt.title("Which group performs better")
plt.legend(title='click', labels=['No', 'Yes'])
total = ab_test_data.groupby('group')['click'].count()
for p in ax.patches:
    height = p.get_height()
    group = p.get_x() + p.get_width() / 2.
    if group < 0.5:
        percentage = height / total.iloc[0] * 100
    else:
        percentage = height / total.iloc[1] * 100
    ax.annotate(f'{percentage:.1f}%', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')
plt.show()


clicks = ab_test_data.groupby("group")["click"].sum()
nobs = ab_test_data.groupby("group")["click"].count()
z_stat, p_val = proportions_ztest([clicks['exp'], clicks['con']], [nobs['exp'], nobs['con']])
print("Z-Statistic:", z_stat)
print("P-Value:", p_val)
ci_con = sm.stats.proportion_confint(clicks['con'], nobs['con'], alpha=0.05, method='wilson')
ci_exp = sm.stats.proportion_confint(clicks['exp'], nobs['exp'], alpha=0.05, method='wilson')
print("CI Control:", ci_con)
print("CI Experiment:", ci_exp)
