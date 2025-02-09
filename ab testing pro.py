import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
ab_test_data=pd.read_csv("ab_test_click_data (1).csv")
print("\n",ab_test_data.head())
print("\n",ab_test_data.describe( ))
print("\n",ab_test_data.groupby("group")["click"].count())
print("\n",ab_test_data.groupby("group")["click"].sum())
palette={0:'yellow', 1:'black'}
ax=sns.countplot(x='group',hue='click',data=ab_test_data,palette=palette)
plt.xlabel("group")
plt.ylabel("count")
plt.title("which is better new or old")
plt.legend(title='click',labels={'no','yes'})
# Calculate the percentages
total = ab_test_data.groupby(['group'])['click'].count()
yes_counts = ab_test_data[ab_test_data['click'] == 1].groupby(['group'])['click'].count()
no_counts = ab_test_data[ab_test_data['click'] == 0].groupby(['group'])['click'].count()

# Annotate the bars with percentages
for p in ax.patches:
    height = p.get_height()
    group = p.get_x() + p.get_width() / 2. - 0.25
    if group == 0:  # First group
        percentage = height / total.iloc[0] * 100
        ax.annotate(f'{percentage:.1f}%',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom')
    else:  # Second group
        percentage = height / total.iloc[1] * 100
        ax.annotate(f'{percentage:.1f}%',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom')
plt.show()
x_con=ab_test_data.groupby("group")["click"].sum().loc["con"]
x_exp=ab_test_data.groupby("group")["click"].sum().loc["exp"]
print("\n number of clicks in con",x_con)
print(" number of clicks in exp",x_exp)