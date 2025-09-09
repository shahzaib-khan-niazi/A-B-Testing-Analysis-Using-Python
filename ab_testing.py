import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
from sklearn.model_selection import train_test_split
import warnings
ab_data = pd.read_csv('ab_test_click_data (1).csv')
print(ab_data)
print("checking duplicates\n",ab_data.duplicated().sum())
print("\nchecking null values\n",ab_data.isnull().sum())
#analysis is purely about overall click-through rates between groups no need of timestamps

ab_data.drop(columns=['timestamp'], inplace=True)
print(ab_data)

print("\ntotal user in each group:\n",ab_data.groupby('group')['click'].count())
print("\ntotal clicks in each group:\n",ab_data.groupby('group')['click'].sum())

print("\noverall mean click:", ab_data['click'].mean())
print("\noverall click variance:", ab_data['click'].var())

print("\nmean click rate per group:\n",ab_data.groupby('group')['click'].mean())
print("\nvariance per group:\n",ab_data.groupby('group')['click'].var())
print("\nstandard deviation per group:\n",ab_data.groupby('group')['click'].std())


total_clicks = ab_data.groupby('group')['click'].sum()
total_clicks.plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Total clicks: Control vs Experiment')
plt.xlabel('Group')
plt.ylabel('Number of clicks')

for i, count in enumerate(total_clicks):
    plt.text(i, count + 5, f'{count}', ha='center', va='bottom')
plt.show()

sns.histplot(data=ab_data, x='click', hue='group', multiple='dodge', bins=2)
plt.title('click Distribution by Group')
plt.show()

pie=ab_data['click'].value_counts()
plt.pie(pie  ,labels=['No Click', 'Click'], autopct='%1.1f%%', startangle=180)
plt.title('overall Click vs No Click')
plt.ylabel('')
plt.show()

ab_data['group_num']=(ab_data['group']=='exp').astype(int)
X = ab_data[['group_num']]
y = ab_data['click']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print('accuracy ',accuracy_score(y_test,y_pred))
print('\nprecision \n',precision_score(y_test,y_pred))
print('\nf1 score \n',f1_score(y_test,y_pred))
print('\nrecall score\n',recall_score(y_test,y_pred))
print(ab_data[['group', 'group_num']].head())

y_pred_prob=model.predict_proba(X_test)[:, 1]
plt.figure(figsize=(8,5))
sns.histplot(y_pred_prob, bins=20, kde=True, color='blue')
plt.title('Distribution of Predicted Probabilities')
plt.xlabel('Predicted Probability of Click')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(y_pred_prob, y_test, alpha=0.3)
plt.title('Predicted Probability vs Actual Click')
plt.xlabel('Predicted Probability')
plt.ylabel('Actual Click (0 or 1)')
plt.show()

y_pred_prob_zero = model.predict_proba(X_test)[:, 0]
plt.figure(figsize=(8,5))
sns.histplot(y_pred_prob_zero, bins=20, kde=True, color='red')
plt.title('Distribution of Predicted Probabilities for Click = 0')
plt.xlabel('Predicted Probability of No Click (0)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(y_pred_prob_zero, y_test, alpha=0.3)
plt.title('Predicted Probability (0) vs Actual Click')
plt.xlabel('Predicted Probability of No Click (0)')
plt.ylabel('Actual Click (0 or 1)')
plt.show()

group_clicks = ab_data.groupby('group')['click'].mean()
diff = group_clicks['exp'] - group_clicks['con']
lift = diff / group_clicks['con']
print(f'\nDifference:\n' ,diff)
print(f'\nLift: \n',lift)

con_clicks = ab_data[ab_data['group'] == 'con']['click'].sum()
exp_clicks = ab_data[ab_data['group'] == 'exp']['click'].sum()
con_total = ab_data[ab_data['group'] == 'con']['click'].count()
exp_total = ab_data[ab_data['group'] == 'exp']['click'].count()

odds_con = con_clicks / (con_total - con_clicks)
odds_exp = exp_clicks / (exp_total - exp_clicks)
odds_ratio = odds_exp / odds_con
print(f"\nodds ratio (exp vs con):",odds_ratio)

print("\ncontingency table (click vs group):")
print(pd.crosstab(ab_data['group'], ab_data['click']))

# optional: hide the feature name warning
warnings.filterwarnings("ignore", category=UserWarning)

X_plot = np.linspace(0, 1, 100).reshape(-1, 1)  # values between 0 and 1
y_prob = model.predict_proba(X_plot)[:, 1]
plt.plot(X_plot, y_prob, color="blue", label="Logistic Curve")
plt.scatter(X, y, alpha=0.2, color="red", label="Actual data")
plt.xlabel("Group (0=Control, 1=Experiment)")
plt.ylabel("Probability of Click")
plt.title("Logistic Regression on A/B Test Data")
plt.legend()
plt.show()

if group_clicks['exp'] > group_clicks['con']:
    print("\nThe Experiment group (exp) is better with a higher click-through rate "
          f"({group_clicks['exp']:.2%}) compared to Control (con) ({group_clicks['con']:.2%}).")
else:
    print("\nThe Control group (con) performed better with a higher click-through rate "
          f"({group_clicks['con']:.2%}) compared to Experiment (exp) ({group_clicks['exp']:.2%}).")
