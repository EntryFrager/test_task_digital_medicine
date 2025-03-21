import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#------------------------------------------------------Decision Rule Without ML------------------------------------------------------

def custom_rule(df):
    """
        Rule to predict Churn using no more than 3 conditions.
    """

    return ((df['Total day charge'] > 45) & 
            (df['Customer service calls'] > 3) & 
            (df['Total day minutes'] > 240)).astype(int)

train_df = pd.read_csv('./data/table_data_train.csv')
test_df = pd.read_csv('./data/table_data_test.csv')

df = pd.concat([train_df, test_df])
print(df.shape)

#------------------------------------------------Check data types and missing values-------------------------------------------------

print(df.head())
print(df.info())
print(df.isnull().sum())

#----------------------------------------------------Numerical Features Analysis-----------------------------------------------------

features = ['Total day minutes', 'Total intl charge', 'Customer service calls']
churn_groups_mean = df.groupby('Churn')[features].mean()

print(churn_groups_mean)

#----------------------------------------------------Distribution Visualization------------------------------------------------------

churn_null = df[df['Churn'] == 0]['Customer service calls']
churn_one  = df[df['Churn'] == 1]['Customer service calls']

print("\nChurn = 0")
print(churn_null.quantile([0.75, 0.85, 0.9]), '\n')

print("Churn = 1")
print(churn_one.quantile([0.75, 0.85, 0.9]), '\n')

churn_0 = df[df['Churn'] == 0]['Total day charge']
churn_1 = df[df['Churn'] == 1]['Total day charge']

print("Churn = 0\n")
print(churn_0.quantile([0.75, 0.85, 0.9]), '\n')

print("Churn = 1\n")
print(churn_1.quantile([0.75, 0.85, 0.9]), '\n')

churn_0 = df[df['Churn'] == 0]['Total day minutes']
churn_1 = df[df['Churn'] == 1]['Total day minutes']

print("Churn = 0\n")
print(churn_0.quantile([0.75, 0.85, 0.9]), '\n')

print("Churn = 1\n")
print(churn_1.quantile([0.75, 0.85, 0.9]), '\n')

# A histogram of `Customer service calls` for `Churn=0` and `Churn=1`
plt.hist(df[df['Churn'] == 0]['Customer service calls'], alpha = 0.5, label = 'Churn = 0')
plt.hist(df[df['Churn'] == 1]['Customer service calls'], alpha = 0.5, label = 'Churn = 1')
plt.xlabel('Customer service calls')
plt.ylabel('Quantity of customers')
plt.axvline(x = 3)
plt.legend()
plt.show()

# A boxplot for `Total day minutes` segmented by Churn
sns.boxplot(x = 'Churn', y = 'Total day minutes', data = df)
plt.xlabel('Churn')
plt.ylabel('Total day minutes')
plt.show()

# A bar chart showing the churn rate for `International plan` (Yes/No)
churn_rate = df.groupby('International plan')['Churn'].mean()
churn_rate.plot(kind = 'bar')
plt.xlabel('International plan')
plt.ylabel('Churn rate')
plt.show()

#--------------------------------------------------------Correlation Analysis--------------------------------------------------------

corr_matrix = df.corr(numeric_only = True)
corr_matrix_churn = corr_matrix['Churn'].abs().sort_values(ascending = False)
print(corr_matrix_churn.head(4))

#--------------------------------------------------Checking the accuracy of our rule-------------------------------------------------

test_df['Predicted'] = custom_rule(test_df)
accuracy = (test_df['Predicted'] == test_df['Churn']).mean() * 100
print(f"\nAccuracy: {accuracy:.2f}%")

assert accuracy >= 0.82