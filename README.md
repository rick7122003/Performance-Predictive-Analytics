# Capstone AI & Machine Learning: Sales prediction

<img width="298" height="31" alt="image" src="https://github.com/user-attachments/assets/ad7b6eb4-5ab6-4f14-be25-4086fb8d387d" />


Data Cleaning and Data preparation


Summary and Findings: Data Overview:

The dataset originally contained 3215 rows and 45 columns.

Duplicate columns (those with .1 suffix) were successfully identified and removed, reducing the column count to 39, which streamlines the dataset and eliminates redundancy.

Data Type Conversions:

Date Fields: Close Date, Created Date, and Today columns were successfully converted from object to datetime format. This allows for proper chronological analysis, such as calculating deal duration or tracking sales trends over time.

Monetary and Percentage Fields:

Sales, Weekly Sales, and Current Quarter Quota were converted from object to float, with '$', ',', 'K', and 'M' characters handled appropriately for accurate numerical calculations.

Discount Granted was converted from object to float by removing the '%' sign and dividing by 100, making it ready for quantitative analysis.

Boolean Fields: Closed and Won columns were converted from boolean to integer (0 for False and 1 for True). This makes these columns usable in numerical aggregations and modeling.

Key Numerical Insights (from df.describe()):

Amount: The Amount field shows a wide range, with a minimum of 
100,000,000.0. The average Amount is approximately 
7,532,130.65) and the large difference between the mean and the 75th percentile ($13,952.0) indicate a highly skewed distribution with a few very large deals. This suggests potential outliers or a Pareto distribution where a small number of deals account for a significant portion of the total sales.

Sales: The Sales column has a similar distribution to Amount, with values ranging from 
100,000,000. The mean Sales is around $442,166.75, which is lower than the average Amount. This discrepancy might indicate that not all Amount values are converted into Sales (e.g., due to lost opportunities or differences in reporting).

Discount Granted: The average Discount Granted is approximately 17.5%, with a maximum of 60.0% and a minimum of 0.0%. This metric can be useful for understanding pricing strategies and their impact on sales.

Opportunity Quantity: The Opportunity Quantity ranges from 1.0 to 100.0, with an average of about 15.68 units per opportunity.

Unit Price: The Unit Price has a median of 400 and a mean of 455.51, with a minimum of 1 and a maximum of 800. This variability suggests different product categories or pricing models.

Key Categorical Insights (from value_counts()):

Account Type: "Gold" is the most prevalent account type, followed by "Platinum". This might indicate a tiered customer segmentation where Gold accounts are the most common, but Platinum accounts could represent higher-value clients.

Industry: "Technology" and "Healthcare" are the most frequent industries, suggesting these are key target sectors for the business. This information can be used to tailor sales and marketing efforts.

Opportunity Type: "Software" and "Maintenance" are the dominant opportunity types, indicating these are the primary offerings or deal categories.

Stage: "Closed Won" and "Closed Lost" are the most common stages, which is expected as they represent the final outcomes of opportunities. A healthy pipeline would also show a good mix of "Qualification," "Needs Analysis," and "Proposal/Price Quote" stages.

Current Quarter?: The majority of records are from "Other Quarters", which indicates that the dataset contains historical data not primarily focused on the current quarter's performance.

In conclusion, the data is now cleaned and prepared for further in-depth sales analysis. The initial insights gained from these summary statistics highlight areas for further investigation, such as the distribution of high-value deals, the relationship between discounts and sales, and the performance of different account types or industries. The cleaned data frame is stored in df.


<img width="976" height="874" alt="image" src="https://github.com/user-attachments/assets/147e6cf8-7138-4a55-bd52-f29f4152e390" />


import pandas as pd

# Load the dataset
df = pd.read_csv('salesforce data.csv')

df.info()

df.describe()

<img width="268" height="594" alt="image" src="https://github.com/user-attachments/assets/c9c76a43-7491-41b0-82e4-9d3f48145829" />

<img width="737" height="241" alt="image" src="https://github.com/user-attachments/assets/1305db4b-b92e-4998-9712-703bc5432059" />


<img width="573" height="412" alt="image" src="https://github.com/user-attachments/assets/c7edc450-80a3-4b89-bc9e-77b7bd5e4cbe" />


<img width="422" height="543" alt="image" src="https://github.com/user-attachments/assets/b1d88ba3-3824-49da-acee-061e2340c182" />


<img width="738" height="563" alt="image" src="https://github.com/user-attachments/assets/9e381105-ca90-4994-9962-7eb000292aa0" />


<img width="458" height="731" alt="image" src="https://github.com/user-attachments/assets/ab8a57c7-dcd0-4ba2-9fc5-d3e065705ae7" />


<img width="248" height="277" alt="image" src="https://github.com/user-attachments/assets/aebdb456-e592-4195-8fe7-e70d17765785" />



EDA analysis and visualize charts & findings, report key take away findings.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'df' is your cleaned and prepared DataFrame from previous steps

# --- Additional EDA with Chart Visualization ---

# 1. Scatter plot: Sales vs. Amount, colored by Discount Granted
# This helps visualize how discounts might influence the relationship between opportunity amount and actual sales.
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Amount', y='Sales', hue='Discount Granted', data=df, palette='viridis', alpha=0.7, size='Discount Granted', sizes=(20, 400))
plt.title('Sales vs. Amount (Colored by Discount Granted)')
plt.xlabel('Opportunity Amount ($)')
plt.ylabel('Sales ($)')
plt.xscale('log') # Use log scale for better visualization due to wide range of values
plt.yscale('log') # Use log scale for better visualization due to wide range of values
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 2. Stacked Bar Chart: Win Rate by Account Type
# This shows the proportion of 'Won' vs. 'Lost' opportunities for each account type.
account_type_stage_counts = df.groupby(['Account Type', 'Won']).size().unstack(fill_value=0)
account_type_stage_percent = account_type_stage_counts.apply(lambda x: x / x.sum(), axis=1)

account_type_stage_percent.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Paired')
plt.title('Opportunity Win/Loss Rate by Account Type')
plt.xlabel('Account Type')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Won (1=True, 0=False)', labels=['Lost', 'Won'])
plt.tight_layout()
plt.show()

# 3. Bar Chart: Total Sales by Billing Region
# Understand which geographic regions contribute most to sales.
sales_by_billing_region = df.groupby('Billing Region')['Sales'].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 7))
sns.barplot(x=sales_by_billing_region.index, y=sales_by_billing_region.values, palette='coolwarm')
plt.title('Total Sales by Billing Region')
plt.xlabel('Billing Region')
plt.ylabel('Total Sales ($)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 4. Bar Chart: Top 10 Product Names by Number of Opportunities
# Identify the most frequently occurring products in opportunities.
top_products = df['Product Name'].value_counts().nlargest(10)
plt.figure(figsize=(12, 7))
sns.barplot(x=top_products.index, y=top_products.values, palette='crest')
plt.title('Top 10 Product Names by Number of Opportunities')
plt.xlabel('Product Name')
plt.ylabel('Number of Opportunities')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 5. Box Plot: Sales Distribution by Industry
# Visualize the distribution of sales values for different industries, showing median, quartiles, and outliers.
plt.figure(figsize=(14, 8))
sns.boxplot(x='Industry', y='Sales', data=df, palette='Set3')
plt.title('Sales Distribution by Industry')
plt.xlabel('Industry')
plt.ylabel('Sales ($)')
plt.xticks(rotation=45, ha='right')
plt.yscale('log') # Use log scale if sales values have a wide range
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


<img width="513" height="654" alt="image" src="https://github.com/user-attachments/assets/93374004-76b8-4589-a986-253ee33db61f" />


<img width="550" height="318" alt="image" src="https://github.com/user-attachments/assets/a8735783-b7ab-4192-b76a-5137ea134db9" />


<img width="549" height="314" alt="image" src="https://github.com/user-attachments/assets/939d2335-1764-4e3d-b15a-7f3460a8430c" />



<img width="546" height="316" alt="image" src="https://github.com/user-attachments/assets/7257f795-d4f3-403d-a88c-d3d5bcc1337a" />



Summary of Findings and Key Takeaways: Sales vs. Amount (Colored by Discount Granted):

The scatter plot reveals a strong positive correlation between Opportunity Amount and Sales, which is expected. Most points lie along the y=x line, indicating that the closed sales value is often very close to the initial opportunity amount.

The Discount Granted does not show a clear pattern of severely reducing sales relative to the initial amount. Both high and low discounts are present across various Amount and Sales ranges. There might be cases where larger discounts are given for higher value deals to close them, but it doesn't appear to systematically pull sales significantly below the potential amount.

Key Takeaway: The sales team generally closes deals close to their original proposed amount. While discounts are given, they don't seem to drastically alter the Amount to Sales conversion on a large scale. Further investigation into specific high-discount scenarios could reveal their impact on profitability.

Opportunity Win/Loss Rate by Account Type:

Gold and Platinum accounts generally show a higher proportion of "Won" opportunities compared to "Lost" ones. This is a positive sign, indicating effective engagement with high-tier customers.

"Silver" and "Bronze" accounts, while having fewer overall opportunities, appear to have a lower win rate compared to "Gold" and "Platinum."

Key Takeaway: Strategic focus on "Gold" and "Platinum" accounts is paying off in terms of win rates. Consider analyzing the sales strategies or challenges specific to "Silver" and "Bronze" accounts to improve their win rates, or re-evaluate the effort allocation if these tiers are less profitable.

Total Sales by Billing Region:

The "West" region significantly outperforms other regions in terms of total sales, followed by "Southwest" and "Northeast."

"Central," "Southeast," and "Midwest" regions contribute substantially less to overall sales.

Key Takeaway: The "West" region is a primary revenue driver. Investigate the success factors in the "West" (e.g., market density, sales team strength, specific industries) and consider replicating successful strategies in underperforming regions. Resource allocation and sales targets might need to be adjusted based on regional performance.

Top 10 Product Names by Number of Opportunities:

"MOL Standard" is by far the most frequently appearing product in opportunities, indicating its widespread presence in deals.

Other products like "MOL Mobile" and "MOL Express" also appear frequently but significantly less than "MOL Standard."

Key Takeaway: "MOL Standard" is a flagship product or a common component in many deals. Understanding its sales cycle, customer satisfaction, and cross-selling potential with other products is crucial. For other products, evaluate if their lower frequency is due to market demand or less focused sales efforts.

Sales Distribution by Industry:

Industries like "Technology," "Healthcare," and "Financial Services" show a wide range of sales, including some very high-value deals (indicated by longer tails or outliers in the box plots). This suggests these industries are capable of generating significant revenue.

Other industries might have a tighter distribution of sales values, indicating more consistent but potentially lower average deal sizes.

Key Takeaway: Focus sales and marketing efforts on industries that consistently yield high-value opportunities. Identify the characteristics of high-value deals within these industries to optimize sales strategies. For industries with lower average sales, consider volume-based strategies or re-evaluating the effort-to-return ratio.



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming 'df' is your cleaned and prepared DataFrame from previous steps

# --- EDA using Pairplot and Correlation Matrix ---

# Select relevant numerical features for analysis
# Exclude 'Sales_Category' as it's a derived target, not an original numerical feature for this context.
# 'Sales' and 'Amount' are kept as their relationship is key, along with other quantitative measures.
numerical_features_for_eda = [
    'Amount', 'Sales', 'Opportunity Quantity', 'Discount Granted',
    'Customer Count', 'Unit Price', 'Software Sold', 'Transactions',
    'Days Left before EOQ'
]

# Ensure only existing columns are selected
numerical_features_for_eda = [col for col in numerical_features_for_eda if col in df.columns]

# Drop rows with NaN in these specific columns for accurate pairplot and correlation
# It's better to handle missing values at the initial data cleaning stage, but for this specific EDA subset,
# we ensure no NaNs affect the plots.
df_eda_numerical = df[numerical_features_for_eda].dropna()

# 1. Pairplot for selected numerical features
# This generates scatter plots for all pairwise relationships and histograms for individual features.
print("Generating Pairplot...")
sns.pairplot(df_eda_numerical)
plt.suptitle('Pairwise Relationships of Key Numerical Features', y=1.02) # Adjust suptitle position
plt.show()
print("Pairplot displayed.")


# 2. Correlation Matrix for selected numerical features
print("\nGenerating Correlation Matrix...")
correlation_matrix = df_eda_numerical.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5,
            cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Matrix of Key Numerical Features')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
print("Correlation Matrix displayed.")



<img width="740" height="774" alt="image" src="https://github.com/user-attachments/assets/a98a80bb-abf5-4c64-b3a5-31694fb283a9" />



<img width="738" height="677" alt="image" src="https://github.com/user-attachments/assets/f91cb19e-e56e-42ae-86cf-cba8a159629f" />



from IPython.display import HTML
HTML('<div style="font-size: 30px; font-weight: bold;">'
     'Evaluation of Machine Learning Models'
     '</div>')

Evaluation of Machine Learning Models

Choices of models for evaluation:
Random Forest Regressor
Gradient Boosting Regressor (LightGBM)
Logistic Regression model
Support Vector Regressor (SVR)
K-Nearest Neighbors (KNN) Regressor



Random Forest Regressor
Apply Random Forest Regressor model. I like to predict a categorical outcome of "Sales" versus "Industry", "Product Name", "Opportunity Type", "Amount", "Opportunity Quantity", "Discount Granted", "Customer Count", "Account Type", "Billing Region", "Stage". Include most suitable 2 chartplot for visualization. Validate the model. Report model based on performance metric of accuracy, precision, recall and F1 score. Report findings on suitability of the model.

Random Forest Classifier Model for Categorical Sales Outcome Prediction (Expanded Features) To predict a categorical outcome, the Sales variable was transformed into Sales_Category ('High Sales' / 'Low Sales') based on its median. The Random Forest Classifier model was then trained using an expanded set of features.

Model Performance Metrics: Accuracy: 1.0000

Precision (High Sales): 1.0000

Recall (High Sales): 1.0000

F1-Score (High Sales): 1.0000

Findings on Suitability of the Model: Perfect Performance Metrics - Due to Data Leakage:

The model achieved perfect scores (Accuracy, Precision, Recall, and F1-Score all 1.0000). While this might seem ideal, in a real-world scenario, such results are a strong indication of data leakage.

The primary cause of this data leakage is the inclusion of the Amount feature in your input features. Your Sales_Category target variable is derived directly from the Sales column, and Sales is almost perfectly correlated with Amount (especially for closed opportunities). By giving the model Amount as a feature, you are essentially providing it with the answer directly, making the prediction trivial.

Other features like Stage (if "Closed Won" or "Closed Lost" stages are directly correlated with Sales being above or below the median) could also contribute to this leakage.

Model Suitability (Under Ideal Circumstances, without Leakage):

Random Forest Classifier is a highly suitable model for classification tasks. It is robust, handles various data types (numerical and one-hot encoded categorical) well, and is less prone to overfitting than individual decision trees.

If features were truly independent predictors (i.e., no data leakage), Random Forest would likely achieve very high, but not perfect, accuracy due to its ensemble nature and ability to capture complex non-linear relationships.

Recommendations for Building a Truly Predictive Model:

Eliminate Data Leakage: The most crucial step is to remove the Amount feature from your features_for_classification. If your goal is to predict Sales_Category before the final sale amount is known, then Amount is an outcome, not a predictor.

Re-evaluate Stage: If Stage values like "Closed Won" or "Closed Lost" are used, and they directly map to 'High Sales' or 'Low Sales', consider if these should be used as features or if you want to predict based on earlier, pre-closure stages.

Focus on Leading Indicators: Once leakage features are removed, the model will be forced to learn from true leading indicators. The performance metrics will drop significantly, but they will then accurately reflect the model's ability to predict Sales_Category from genuinely available prior information.

Further Feature Engineering: With non-leaking features, you can explore creating new features (e.g., ratios, aggregations, time-based features) or acquiring external data to improve predictive power.

Hyperparameter Tuning: Once a realistic performance baseline is established, fine-tuning Random Forest's hyperparameters (e.g., n_estimators, max_depth, min_samples_split, min_samples_leaf) can further optimize its performance.

In conclusion, while the Random Forest Classifier demonstrated perfect performance, this is a strong sign of data leakage, primarily due to the Amount feature directly revealing the target. To develop a truly valuable and generalizable predictive model for Sales_Category, it is essential to cleanse the feature set of any variables that are direct outcomes or direct derivations of the target.

Chart Visualizations: Amount Distribution by Sales Category:

This box plot visually confirms why the model achieved perfect scores. It clearly shows a distinct separation in Amount values between 'Low Sales' and 'High Sales' categories. This makes it trivial for a model with Amount as a feature to classify.

Sales Category Distribution by Account Type:

This bar chart illustrates the distribution of 'High Sales' and 'Low Sales' opportunities across different 'Account Type' categories. This visualization remains useful for understanding the categorical relationships within your data, independent of the data leakage issue.




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier # Import RandomForestClassifier
from sklearn.preprocessing import StandardScaler # For feature scaling
import numpy as np

# 1. Categorize 'Sales' into 'Low Sales' and 'High Sales' based on median
median_sales = df['Sales'].median()
df['Sales_Category'] = df['Sales'].apply(lambda x: 'High Sales' if x > median_sales else 'Low Sales')

# Define target and features with the expanded list
target_category = 'Sales_Category'
features_for_classification = [
    'Industry', 'Product Name', 'Opportunity Type', 'Amount', 'Opportunity Quantity',
    'Discount Granted', 'Customer Count', 'Account Type', 'Billing Region', 'Stage'
]

# Filter DataFrame to include only the relevant columns for this analysis
df_clf = df[features_for_classification + [target_category]].copy()

# Handle missing values in the selected features
# Identify numerical and categorical features in the selected subset
numerical_features_clf = df_clf[features_for_classification].select_dtypes(include=np.number).columns
categorical_features_clf = df_clf[features_for_classification].select_dtypes(include='object').columns

# Impute numerical missing values with the median
for col in numerical_features_clf:
    if df_clf[col].isnull().sum() > 0:
        df_clf[col] = df_clf[col].fillna(df_clf[col].median())

# Impute categorical missing values with the mode
for col in categorical_features_clf:
    if df_clf[col].isnull().sum() > 0:
        df_clf[col] = df_clf[col].fillna(df_clf[col].mode()[0])

# Drop rows where the target variable itself might be missing (should be handled by previous steps)
df_clf.dropna(subset=[target_category], inplace=True)


# Separate features (X_clf) and target (y_clf)
X_clf = df_clf[features_for_classification]
y_clf = df_clf[target_category]

# Apply One-Hot Encoding to categorical features
X_clf_encoded = pd.get_dummies(X_clf, columns=categorical_features_clf, drop_first=True)

# Apply Feature Scaling to numerical features after one-hot encoding
# Identify which columns in the encoded dataframe are original numerical features
numerical_cols_after_encoding = [col for col in X_clf_encoded.columns if col in numerical_features_clf]

scaler = StandardScaler()
X_clf_encoded[numerical_cols_after_encoding] = scaler.fit_transform(X_clf_encoded[numerical_cols_after_encoding])


# Split the data into training and testing sets with stratification
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf_encoded, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Initialize and train the Random Forest Classifier model
# Using default parameters for simplicity.
model_rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10) # Limiting max_depth for simplicity
model_rf.fit(X_train_clf, y_train_clf)

# Make predictions on the test set
y_pred_rf = model_rf.predict(X_test_clf)

# Evaluate the model
accuracy = accuracy_score(y_test_clf, y_pred_rf)
precision = precision_score(y_test_clf, y_pred_rf, pos_label='High Sales')
recall = recall_score(y_test_clf, y_pred_rf, pos_label='High Sales')
f1 = f1_score(y_test_clf, y_pred_rf, pos_label='High Sales')

print("--- Random Forest Classifier Model Performance Metrics (Expanded Features) ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (High Sales): {precision:.4f}")
print(f"Recall (High Sales): {recall:.4f}")
print(f"F1-Score (High Sales): {f1:.4f}")

# --- Visualizations (using expanded features) ---

# 1. Box Plot: Sales_Category vs Amount
plt.figure(figsize=(10, 6))
sns.boxplot(x='Sales_Category', y='Amount', data=df_clf, palette='pastel')
plt.title('Amount Distribution by Sales Category')
plt.xlabel('Sales Category')
plt.ylabel('Amount ($)')
plt.yscale('log') # Use log scale due to wide range of Amount values
plt.tight_layout()
plt.show()

# 2. Bar Chart: Sales_Category distribution by Account Type
plt.figure(figsize=(10, 6))
sns.countplot(x='Account Type', hue='Sales_Category', data=df_clf, palette='coolwarm')
plt.title('Sales Category Distribution by Account Type')
plt.xlabel('Account Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sales Category')
plt.tight_layout()
plt.show()


--- Random Forest Classifier Model Performance Metrics (Expanded Features) ---
Accuracy: 1.0000
Precision (High Sales): 1.0000
Recall (High Sales): 1.0000
F1-Score (High Sales): 1.0000


<img width="704" height="784" alt="image" src="https://github.com/user-attachments/assets/3079dbc7-86c9-470f-9791-05f0d463f688" />


Apply Gradient Boosting Regressor (XGBoost) model.
Apply Gradient Boosting Regressor (LightGBM) model. I like to predict a categorical outcome of "Sales" versus "Industry", "Product Name", "Opportunity Type", "Amount", "Opportunity Quantity", "Discount Granted", "Customer Count", "Account Type", "Billing Region", "Stage". Include most suitable 2 chartplot for visualization. Validate the model. Report model based on performance metric of accuracy, precision, recall and F1 score. Report findings on suitability of the model.

LightGBM Classifier Model for Categorical Sales Outcome Prediction (Expanded Features) Please Note: Due to the lightgbm library not being detected in the current environment, the model's actual performance metrics could not be computed in this turn. The following analysis of suitability and expected performance is based on the general characteristics of LightGBM and the significant performance improvements observed with other models (SVC, Logistic Regression) when using the expanded feature set.

Expected Model Performance Metrics: Given the strong performance of other models (like SVC and Logistic Regression, both achieving around 88-90% accuracy) with the expanded feature set, it is expected that the LightGBM Classifier would also achieve very high performance metrics.

Expected Accuracy: Likely in the range of 0.89 - 0.93.

Expected Precision, Recall, F1-Score (High Sales): Also expected to be high, likely in the range of 0.85 - 0.95.

Findings on Suitability of the Model: Strong Predictive Power (Expected):

LightGBM is an extremely efficient and highly performant gradient boosting framework. Its speed and accuracy, even on large datasets, make it a top choice for many classification problems. With the inclusion of highly predictive numerical features like Amount and other relevant categorical features, LightGBM is expected to achieve excellent accuracy and other metrics in classifying Sales_Category.

This model, similar to SVC and Logistic Regression (with expanded features), should demonstrate that access to more informative features drastically improves predictive capabilities, moving from near-random guessing to highly accurate predictions.

Suitability with Expanded Features:

LightGBM is highly suitable for this classification task. It is designed for efficiency and handles various data types well. While we are using one-hot encoding here, LightGBM can also be configured to handle categorical features natively, which can sometimes further improve performance.

Robustness: LightGBM incorporates various techniques to prevent overfitting, making it a robust choice for complex datasets.

Data Leakage Warning:

Crucially, it is highly probable that the model will achieve near-perfect or perfect performance (e.g., 1.0 for all metrics) due to the inclusion of the Amount feature. As discussed previously, Sales_Category is derived directly from Sales, and Sales is fundamentally determined by Amount for closed opportunities. Therefore, including Amount as a feature means the model is essentially being given the answer, leading to data leakage.

If you desire a model that predicts Sales_Category based on features available before the final amount is known (i.e., true leading indicators), then Amount (and potentially Stage if it directly implies the outcome) should be removed from your feature set.

Limitations and Further Considerations:

Interpretability: While LightGBM can provide feature importances, the "why" behind specific predictions can be less intuitive than simpler models.

Hyperparameter Tuning: Fine-tuning LightGBM's hyperparameters (e.g., num_leaves, learning_rate, n_estimators, feature_fraction, bagging_fraction) could further optimize performance for your specific dataset.

In summary, the LightGBM Classifier is an excellent and highly suitable choice for predicting the Sales_Category when provided with the expanded set of features. It is expected to yield very high performance metrics. However, the presence of the Amount feature in the input is likely to cause data leakage, resulting in artificially perfect (or near-perfect) scores. For a truly predictive model based on leading indicators, consider carefully refining your feature set to remove such outcome-derived variables.

Chart Visualizations: Amount Distribution by Sales Category:

This box plot clearly illustrates that 'High Sales' opportunities are associated with significantly higher Amount values than 'Low Sales' opportunities. This visual strongly highlights the predictive power of Amount and why its inclusion can lead to data leakage if Sales_Category is derived from Sales (which is often very close to Amount).

Sales Category Distribution by Account Type:

This bar chart provides insight into the distribution of 'High Sales' and 'Low Sales' across different 'Account Type' categories. It can help identify if certain account types are more likely to result in high-value sales, even if Amount is removed.




!pip install lightgbm


<img width="573" height="73" alt="image" src="https://github.com/user-attachments/assets/bb24b28e-5593-48ed-a7cd-e4c0a50386a0" />




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler # For feature scaling
import lightgbm as lgb # Make sure this library is installed
import numpy as np

# 1. Categorize 'Sales' into 'Low Sales' and 'High Sales' based on median
median_sales = df['Sales'].median()
df['Sales_Category'] = df['Sales'].apply(lambda x: 'High Sales' if x > median_sales else 'Low Sales')

# Define target and features with the expanded list
target_category = 'Sales_Category'
features_for_classification = [
    'Industry', 'Product Name', 'Opportunity Type', 'Amount', 'Opportunity Quantity',
    'Discount Granted', 'Customer Count', 'Account Type', 'Billing Region', 'Stage'
]

# Filter DataFrame to include only the relevant columns for this analysis
df_clf = df[features_for_classification + [target_category]].copy()

# Handle missing values in the selected features
# Identify numerical and categorical features in the selected subset
numerical_features_clf = df_clf[features_for_classification].select_dtypes(include=np.number).columns
categorical_features_clf = df_clf[features_for_classification].select_dtypes(include='object').columns

# Impute numerical missing values with the median
for col in numerical_features_clf:
    if df_clf[col].isnull().sum() > 0:
        df_clf[col] = df_clf[col].fillna(df_clf[col].median())

# Impute categorical missing values with the mode
for col in categorical_features_clf:
    if df_clf[col].isnull().sum() > 0:
        df_clf[col] = df_clf[col].fillna(df_clf[col].mode()[0])

# Drop rows where the target variable itself might be missing (should be handled by previous steps)
df_clf.dropna(subset=[target_category], inplace=True)


# Separate features (X_clf) and target (y_clf)
X_clf = df_clf[features_for_classification]
y_clf = df_clf[target_category]

# Apply One-Hot Encoding to categorical features
X_clf_encoded = pd.get_dummies(X_clf, columns=categorical_features_clf, drop_first=True)

# Apply Feature Scaling to numerical features after one-hot encoding
# Identify which columns in the encoded dataframe are original numerical features
numerical_cols_after_encoding = [col for col in X_clf_encoded.columns if col in numerical_features_clf]

scaler = StandardScaler()
X_clf_encoded[numerical_cols_after_encoding] = scaler.fit_transform(X_clf_encoded[numerical_cols_after_encoding])


# Split the data into training and testing sets with stratification
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf_encoded, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Initialize and train the LightGBM Classifier model
# For classification, we use LGBMClassifier.
# 'objective' is 'binary' for two classes.
# Using default parameters for simplicity.
model_lgbm = lgb.LGBMClassifier(objective='binary', random_state=42, n_estimators=100)
model_lgbm.fit(X_train_clf, y_train_clf)

# Make predictions on the test set
y_pred_lgbm = model_lgbm.predict(X_test_clf)

# Evaluate the model
accuracy = accuracy_score(y_test_clf, y_pred_lgbm)
precision = precision_score(y_test_clf, y_pred_lgbm, pos_label='High Sales')
recall = recall_score(y_test_clf, y_pred_lgbm, pos_label='High Sales')
f1 = f1_score(y_test_clf, y_pred_lgbm, pos_label='High Sales')

print("--- LightGBM Model Performance Metrics (Expanded Features) ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (High Sales): {precision:.4f}")
print(f"Recall (High Sales): {recall:.4f}")
print(f"F1-Score (High Sales): {f1:.4f}")

# --- Visualizations (using expanded features) ---

# 1. Box Plot: Sales_Category vs Amount
plt.figure(figsize=(10, 6))
sns.boxplot(x='Sales_Category', y='Amount', data=df_clf, palette='pastel')
plt.title('Amount Distribution by Sales Category')
plt.xlabel('Sales Category')
plt.ylabel('Amount ($)')
plt.yscale('log') # Use log scale due to wide range of Amount values
plt.tight_layout()
plt.show()

# 2. Bar Chart: Sales_Category distribution by Account Type
plt.figure(figsize=(10, 6))
sns.countplot(x='Account Type', hue='Sales_Category', data=df_clf, palette='coolwarm')
plt.title('Sales Category Distribution by Account Type')
plt.xlabel('Account Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sales Category')
plt.tight_layout()
plt.show()


<img width="514" height="764" alt="image" src="https://github.com/user-attachments/assets/ae3e5683-717b-43de-8ac0-c83aa1f88d9d" />




<img width="340" height="62" alt="image" src="https://github.com/user-attachments/assets/4a955403-3ec0-4cd0-af6b-0113efc81ca2" />



<img width="696" height="796" alt="image" src="https://github.com/user-attachments/assets/07bf46da-557f-47b8-b39d-38c068e78134" />



Apply Logistic Regression model. I like to predict a categorical outcome of "Sales" versus "Industry", "Product Name", "Opportunity Type", "Amount", "Opportunity Quantity", "Discount Granted", "Customer Count", "Account Type", "Billing Region", "Stage". Include most suitable 2 chartplot for visualization. Validate the model. Report model based on performance metric of accuracy, precision, recall and F1 score. Report findings on suitability of the model.

Logistic Regression Model for Categorical Sales Outcome Prediction (Expanded Features) To predict a categorical outcome, the Sales variable was transformed into Sales_Category ('High Sales' / 'Low Sales') based on its median. The Logistic Regression model was then trained using an expanded set of features.

Model Performance Metrics:

Accuracy: 0.8818 Precision (High Sales): 0.8217 Recall (High Sales): 0.9609 F1-Score (High Sales): 0.8859

Findings on Suitability of the Model: Significantly Improved Performance:

The Accuracy has substantially increased to approximately 88.18% with the expanded feature set, a significant improvement over its performance with limited features. This shows the model is now much more effective at classifying sales categories.

The Precision for 'High Sales' is 82.17%, indicating good reliability when predicting 'High Sales'.

The Recall for 'High Sales' is remarkably high at 96.09%, meaning the model effectively identifies most actual 'High Sales' opportunities. This is particularly beneficial if the goal is to capture as many high-value deals as possible.

The F1-Score of 0.8859 represents a strong balance between precision and recall, signifying robust overall performance for the 'High Sales' class.

Suitability with Expanded Features:

Logistic Regression is highly suitable for this binary classification task, especially with the enriched feature set and appropriate preprocessing (like scaling and one-hot encoding). It provides a strong, interpretable baseline and can effectively handle complex relationships when given relevant features.

Impact of Feature Engineering: The considerable improvement in all metrics, compared to when only a few features were used, powerfully demonstrates that including more predictive features like Amount, Opportunity Quantity, and Stage dramatically enhances the model's ability to classify sales outcomes.

Interpretability: A key advantage of Logistic Regression is its interpretability. Once trained, you can examine the coefficients to understand the linear relationship and impact of each feature on the likelihood of an opportunity being 'High Sales'.

Limitations and Further Considerations:

Linearity Assumption: Logistic Regression is a linear model at its core. While it can handle non-linear relationships through feature engineering (e.g., polynomial features, interaction terms), it might not capture highly complex, non-linear patterns as effectively as tree-based ensemble methods (like Random Forest or CatBoost) if such patterns are present and not explicitly engineered.

Overfitting (Potential): While robust, with a very high number of one-hot encoded features, there's always a slight risk of overfitting, especially if regularization (C parameter) isn't tuned. However, the current performance suggests good generalization.

Threshold Tuning: The model outputs probabilities, which are then converted to binary classes using a default threshold (0.5). Depending on business needs (e.g., prioritizing recall over precision), this threshold can be tuned.

In summary, the Logistic Regression model, when provided with a comprehensive set of relevant features, performs very well in predicting whether a sales opportunity will result in 'High Sales' or 'Low Sales'. This model is suitable for the task, offers good performance, and retains the benefit of interpretability. The success strongly validates the importance of proper feature engineering in machine learning.






import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression # Import LogisticRegression
from sklearn.preprocessing import StandardScaler # For feature scaling
import numpy as np

# 1. Categorize 'Sales' into 'Low Sales' and 'High Sales' based on median
median_sales = df['Sales'].median()
df['Sales_Category'] = df['Sales'].apply(lambda x: 'High Sales' if x > median_sales else 'Low Sales')

# Define target and features with the expanded list
target_category = 'Sales_Category'
features_for_classification = [
    'Industry', 'Product Name', 'Opportunity Type', 'Amount', 'Opportunity Quantity',
    'Discount Granted', 'Customer Count', 'Account Type', 'Billing Region', 'Stage'
]

# Filter DataFrame to include only the relevant columns for this analysis
df_clf = df[features_for_classification + [target_category]].copy()

# Handle missing values in the selected features
# Identify numerical and categorical features in the selected subset
numerical_features_clf = df_clf[features_for_classification].select_dtypes(include=np.number).columns
categorical_features_clf = df_clf[features_for_classification].select_dtypes(include='object').columns

# Impute numerical missing values with the median
for col in numerical_features_clf:
    if df_clf[col].isnull().sum() > 0:
        df_clf[col] = df_clf[col].fillna(df_clf[col].median())

# Impute categorical missing values with the mode
for col in categorical_features_clf:
    if df_clf[col].isnull().sum() > 0:
        df_clf[col] = df_clf[col].fillna(df_clf[col].mode()[0])

# Drop rows where the target variable itself might be missing (should be handled by previous steps)
df_clf.dropna(subset=[target_category], inplace=True)


# Separate features (X_clf) and target (y_clf)
X_clf = df_clf[features_for_classification]
y_clf = df_clf[target_category]

# Apply One-Hot Encoding to categorical features
X_clf_encoded = pd.get_dummies(X_clf, columns=categorical_features_clf, drop_first=True)

# Apply Feature Scaling to numerical features after one-hot encoding
# Identify which columns in the encoded dataframe are original numerical features
numerical_cols_after_encoding = [col for col in X_clf_encoded.columns if col in numerical_features_clf]

scaler = StandardScaler()
X_clf_encoded[numerical_cols_after_encoding] = scaler.fit_transform(X_clf_encoded[numerical_cols_after_encoding])


# Split the data into training and testing sets with stratification
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf_encoded, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Initialize and train the Logistic Regression model
# Using 'liblinear' solver for robustness with smaller datasets and potential L1/L2 regularization.
# max_iter increased to ensure convergence for complex models.
model_lr = LogisticRegression(solver='liblinear', random_state=42, max_iter=500)
model_lr.fit(X_train_clf, y_train_clf)

# Make predictions on the test set
y_pred_lr = model_lr.predict(X_test_clf)

# Evaluate the model
accuracy = accuracy_score(y_test_clf, y_pred_lr)
precision = precision_score(y_test_clf, y_pred_lr, pos_label='High Sales')
recall = recall_score(y_test_clf, y_pred_lr, pos_label='High Sales')
f1 = f1_score(y_test_clf, y_pred_lr, pos_label='High Sales')

print("--- Logistic Regression Model Performance Metrics (Expanded Features) ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (High Sales): {precision:.4f}")
print(f"Recall (High Sales): {recall:.4f}")
print(f"F1-Score (High Sales): {f1:.4f}")

# --- Visualizations (using expanded features) ---

# 1. Box Plot: Sales_Category vs Amount
plt.figure(figsize=(10, 6))
sns.boxplot(x='Sales_Category', y='Amount', data=df_clf, palette='pastel')
plt.title('Amount Distribution by Sales Category')
plt.xlabel('Sales Category')
plt.ylabel('Amount ($)')
plt.yscale('log') # Use log scale due to wide range of Amount values
plt.tight_layout()
plt.show()

# 2. Bar Chart: Sales_Category distribution by Account Type
plt.figure(figsize=(10, 6))
sns.countplot(x='Account Type', hue='Sales_Category', data=df_clf, palette='coolwarm')
plt.title('Sales Category Distribution by Account Type')
plt.xlabel('Account Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sales Category')
plt.tight_layout()
plt.show()


<img width="737" height="533" alt="image" src="https://github.com/user-attachments/assets/5a55a100-a52e-43ed-8d90-e8742a2a6190" />


<img width="677" height="399" alt="image" src="https://github.com/user-attachments/assets/681fdf89-06b8-42aa-a84f-37842368abba" />



Apply Support Vector Regressor (SVR) model. I like to predict a categorical outcome of "Sales" versus "Industry", "Product Name", "Opportunity Type", "Amount", "Opportunity Quantity", "Discount Granted", "Customer Count", "Account Type", "Billing Region", "Stage". Include most suitable 2 chartplot for visualization. Validate the model. Report model based on performance metric of accuracy, precision, recall and F1 score. Report findings on suitability of the model.

Support Vector Classifier (SVC) Model for Categorical Sales Outcome Prediction (Expanded Features) To predict a categorical outcome, the Sales variable was transformed into Sales_Category ('High Sales' / 'Low Sales') based on its median. The Support Vector Classifier (SVC) model, with an RBF kernel, was then trained using an expanded set of features.

Model Performance Metrics:

Accuracy: 0.8958 Precision (High Sales): 0.8315 Recall (High Sales): 0.9805 F1-Score (High Sales): 0.8999

Findings on Suitability of the Model: Significantly Improved Performance:

The Accuracy has dramatically increased to approximately 89.58%, which is a substantial improvement over previous models that used a limited feature set (e.g., 50-60% accuracy). This indicates that the model is now much better at correctly classifying sales as 'High Sales' or 'Low Sales'.

The Precision for 'High Sales' is 83.15%, meaning that when the model predicts 'High Sales', it is correct over 83% of the time.

The Recall for 'High Sales' is exceptionally high at 98.05%. This signifies that the model is highly effective at identifying nearly all actual 'High Sales' opportunities. This is particularly valuable if the business goal is to ensure that very few 'High Sales' opportunities are missed.

The F1-Score of 0.8999 reflects a strong balance between precision and recall for the 'High Sales' class, indicating robust overall performance.

Suitability with Expanded Features:

SVC is highly suitable for this classification task, especially with the enriched feature set. The RBF kernel allows it to capture complex, non-linear decision boundaries between 'High Sales' and 'Low Sales' in the high-dimensional feature space created by one-hot encoding.

Impact of Feature Engineering: The substantial improvement in performance metrics directly demonstrates the critical role of incorporating more relevant and predictive features. Features like Amount, Opportunity Quantity, and Discount Granted provide a much stronger signal for sales outcomes than just categorical identifiers alone.

Robustness: The model's strong performance on unseen test data suggests good generalization ability.

Limitations and Further Considerations:

Computational Cost: SVC, especially with non-linear kernels like RBF, can be computationally intensive and slower to train on very large datasets compared to tree-based methods like LightGBM or XGBoost.

Hyperparameter Tuning: While the default RBF kernel performed very well, further fine-tuning of SVC hyperparameters (e.g., C for regularization, gamma for kernel width) could potentially yield even marginal improvements or optimize for specific business objectives (e.g., higher precision if false positives are very costly).

Interpretability: SVC models are often considered "black boxes" compared to simpler models like Logistic Regression or tree-based models (which can provide feature importance). Understanding why a particular prediction is made can be challenging.

In summary, the SVC model, leveraging an expanded and more informative set of features (including Amount, Opportunity Quantity, etc.), has achieved a very strong predictive performance for classifying sales as 'High Sales' or 'Low Sales'. This model is highly suitable for the task, validating that rich feature engineering is key to success in machine learning applications.


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC # Import Support Vector Classifier
from sklearn.preprocessing import StandardScaler # For feature scaling
import numpy as np

# 1. Categorize 'Sales' into 'Low Sales' and 'High Sales' based on median
median_sales = df['Sales'].median()
df['Sales_Category'] = df['Sales'].apply(lambda x: 'High Sales' if x > median_sales else 'Low Sales')

# Define target and features with the expanded list
target_category = 'Sales_Category'
features_for_classification = [
    'Industry', 'Product Name', 'Opportunity Type', 'Amount', 'Opportunity Quantity',
    'Discount Granted', 'Customer Count', 'Account Type', 'Billing Region', 'Stage'
]

# Filter DataFrame to include only the relevant columns for this analysis
df_clf = df[features_for_classification + [target_category]].copy()

# Handle missing values in the selected features
# Identify numerical and categorical features in the selected subset
numerical_features_clf = df_clf[features_for_classification].select_dtypes(include=np.number).columns
categorical_features_clf = df_clf[features_for_classification].select_dtypes(include='object').columns

# Impute numerical missing values with the median
for col in numerical_features_clf:
    if df_clf[col].isnull().sum() > 0:
        df_clf[col] = df_clf[col].fillna(df_clf[col].median())

# Impute categorical missing values with the mode
for col in categorical_features_clf:
    if df_clf[col].isnull().sum() > 0:
        df_clf[col] = df_clf[col].fillna(df_clf[col].mode()[0])

# Drop rows where the target variable itself might be missing (should be handled by previous steps)
df_clf.dropna(subset=[target_category], inplace=True)


# Separate features (X_clf) and target (y_clf)
X_clf = df_clf[features_for_classification]
y_clf = df_clf[target_category]

# Apply One-Hot Encoding to categorical features
X_clf_encoded = pd.get_dummies(X_clf, columns=categorical_features_clf, drop_first=True)

# Apply Feature Scaling to numerical features after one-hot encoding
# Identify which columns in the encoded dataframe are original numerical features
numerical_cols_after_encoding = [col for col in X_clf_encoded.columns if col in numerical_features_clf]

scaler = StandardScaler()
X_clf_encoded[numerical_cols_after_encoding] = scaler.fit_transform(X_clf_encoded[numerical_cols_after_encoding])


# Split the data into training and testing sets with stratification
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf_encoded, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Initialize and train the Support Vector Classifier (SVC) model
# Using an 'rbf' kernel for potentially better performance with non-linear relationships.
# C=1.0 is a default regularization parameter.
model_svc = SVC(kernel='rbf', random_state=42, probability=True) # probability=True for future use cases
model_svc.fit(X_train_clf, y_train_clf)

# Make predictions on the test set
y_pred_svc = model_svc.predict(X_test_clf)

# Evaluate the model
accuracy = accuracy_score(y_test_clf, y_pred_svc)
precision = precision_score(y_test_clf, y_pred_svc, pos_label='High Sales')
recall = recall_score(y_test_clf, y_pred_svc, pos_label='High Sales')
f1 = f1_score(y_test_clf, y_pred_svc, pos_label='High Sales')

print("--- Support Vector Classifier (SVC) Model Performance Metrics (Expanded Features) ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (High Sales): {precision:.4f}")
print(f"Recall (High Sales): {recall:.4f}")
print(f"F1-Score (High Sales): {f1:.4f}")

# --- Visualizations (using expanded features) ---

# 1. Box Plot: Sales_Category vs Amount
plt.figure(figsize=(10, 6))
sns.boxplot(x='Sales_Category', y='Amount', data=df_clf, palette='pastel')
plt.title('Amount Distribution by Sales Category')
plt.xlabel('Sales Category')
plt.ylabel('Amount ($)')
plt.yscale('log') # Use log scale due to wide range of Amount values
plt.tight_layout()
plt.show()

# 2. Bar Chart: Sales_Category distribution by Account Type
plt.figure(figsize=(10, 6))
sns.countplot(x='Account Type', hue='Sales_Category', data=df_clf, palette='coolwarm')
plt.title('Sales Category Distribution by Account Type')
plt.xlabel('Account Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sales Category')
plt.tight_layout()
plt.show()


<img width="739" height="533" alt="image" src="https://github.com/user-attachments/assets/bfcbe454-6848-4428-a768-751de2ff033a" />


<img width="701" height="399" alt="image" src="https://github.com/user-attachments/assets/e404c5f2-57ff-43f2-b7d5-38629ee4fd25" />





Apply K-Nearest Neighbors (KNN) Regressor model.

Apply K-Nearest Neighbors (KNN) Regressor model. I like to predict a categorical outcome of "Sales" versus "Industry", "Product Name", "Opportunity Type", "Amount", "Opportunity Quantity", "Discount Granted", "Customer Count", "Account Type", "Billing Region", "Stage". Include most suitable 2 chartplot for visualization. Validate the model. Report model based on performance metric of accuracy, precision, recall and F1 score. Report findings on suitability of the model.

Model Performance Metrics:

Accuracy: 0.8227 Precision (High Sales): 0.7781 Recall (High Sales): 0.8795 F1-Score (High Sales): 0.8257

Model Performance Comparison:

Random Forest Regressor model

Gradient Boosting Regressor (LightGBM) model

CatBoost (Categorical Boosting) model

Logistic Regression model

Support Vector Regressor (SVR) model.

K-Nearest Neighbors (KNN) Regressor model Accuracy: 0.8227 Precision (High Sales): 0.7781 Recall (High Sales): 0.8795 F1-Score (High Sales): 0.8257


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler # For feature scaling
import numpy as np

# 1. Categorize 'Sales' into 'Low Sales' and 'High Sales' based on median
median_sales = df['Sales'].median()
df['Sales_Category'] = df['Sales'].apply(lambda x: 'High Sales' if x > median_sales else 'Low Sales')

# Define target and features with the expanded list
target_category = 'Sales_Category'
features_for_classification = [
    'Industry', 'Product Name', 'Opportunity Type', 'Amount', 'Opportunity Quantity',
    'Discount Granted', 'Customer Count', 'Account Type', 'Billing Region', 'Stage'
]

# Filter DataFrame to include only the relevant columns for this analysis
df_clf = df[features_for_classification + [target_category]].copy()

# Handle missing values in the selected features
# Identify numerical and categorical features in the selected subset
numerical_features_clf = df_clf[features_for_classification].select_dtypes(include=np.number).columns
categorical_features_clf = df_clf[features_for_classification].select_dtypes(include='object').columns

# Impute numerical missing values with the median
for col in numerical_features_clf:
    if df_clf[col].isnull().sum() > 0:
        df_clf[col] = df_clf[col].fillna(df_clf[col].median())

# Impute categorical missing values with the mode
for col in categorical_features_clf:
    if df_clf[col].isnull().sum() > 0:
        df_clf[col] = df_clf[col].fillna(df_clf[col].mode()[0])

# Drop rows where the target variable itself might be missing (should be handled by previous steps)
df_clf.dropna(subset=[target_category], inplace=True)


# Separate features (X_clf) and target (y_clf)
X_clf = df_clf[features_for_classification]
y_clf = df_clf[target_category]

# Apply One-Hot Encoding to categorical features
X_clf_encoded = pd.get_dummies(X_clf, columns=categorical_features_clf, drop_first=True)

# Apply Feature Scaling to numerical features after one-hot encoding
# Identify which columns in the encoded dataframe are original numerical features
numerical_cols_after_encoding = [col for col in X_clf_encoded.columns if col in numerical_features_clf]

scaler = StandardScaler()
X_clf_encoded[numerical_cols_after_encoding] = scaler.fit_transform(X_clf_encoded[numerical_cols_after_encoding])


# Split the data into training and testing sets with stratification
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf_encoded, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Initialize and train the K-Nearest Neighbors (KNN) Classifier model
# n_neighbors is a tunable hyperparameter; 5 is a common default.
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train_clf, y_train_clf)

# Make predictions on the test set
y_pred_knn = model_knn.predict(X_test_clf)

# Evaluate the model
accuracy = accuracy_score(y_test_clf, y_pred_knn)
precision = precision_score(y_test_clf, y_pred_knn, pos_label='High Sales')
recall = recall_score(y_test_clf, y_pred_knn, pos_label='High Sales')
f1 = f1_score(y_test_clf, y_pred_knn, pos_label='High Sales')

print("--- K-Nearest Neighbors (KNN) Classifier Model Performance Metrics (Expanded Features) ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (High Sales): {precision:.4f}")
print(f"Recall (High Sales): {recall:.4f}")
print(f"F1-Score (High Sales): {f1:.4f}")

# --- Visualizations (using expanded features) ---

# 1. Box Plot: Sales_Category vs Amount
plt.figure(figsize=(10, 6))
sns.boxplot(x='Sales_Category', y='Amount', data=df_clf, palette='pastel')
plt.title('Amount Distribution by Sales Category')
plt.xlabel('Sales Category')
plt.ylabel('Amount ($)')
plt.yscale('log') # Use log scale due to wide range of Amount values
plt.tight_layout()
plt.show()

# 2. Bar Chart: Sales_Category distribution by Account Type
plt.figure(figsize=(10, 6))
sns.countplot(x='Account Type', hue='Sales_Category', data=df_clf, palette='coolwarm')
plt.title('Sales Category Distribution by Account Type')
plt.xlabel('Account Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sales Category')
plt.tight_layout()
plt.show()



<img width="740" height="526" alt="image" src="https://github.com/user-attachments/assets/ec242cf7-968c-4081-8c08-ed0874dec06a" />



<img width="698" height="389" alt="image" src="https://github.com/user-attachments/assets/d68e3bbd-9919-481f-8fa9-9f80493c8f06" />




from IPython.display import HTML
HTML('<div style="font-size: 30px; font-weight: bold;">'
     'Gradio Deployment - Apply Support Vector Regressor (SVR) model - Deploy Huggingface - Gradio interface/APP'
     '</div>')

# Gradio Deployment  #



Gradio Deployment - Apply Support Vector Regressor (SVR) model - Deploy Huggingface - Gradio interface/APP


# Apply Support Vector Regressor (SVR) model #


!pip install huggingface_hub


<img width="737" height="224" alt="image" src="https://github.com/user-attachments/assets/3755781b-46fe-4fc4-951b-f8cb92d92f49" />


!pip install gradio scikit-learn pandas numpy


<img width="743" height="798" alt="image" src="https://github.com/user-attachments/assets/34386d5e-954d-430d-b571-b3bbd6c76f9b" />



<img width="747" height="781" alt="image" src="https://github.com/user-attachments/assets/89e1ab56-5b4a-44b7-ae8f-68cfaa98a9a4" />


 # Support Vector Classifier for categorical outcome , Gradio Deployment in huggingface_hub #


from IPython.display import HTML
HTML('<div style="font-size: 30px; font-weight: bold;">'
     'Support Vector Classifier for categorical outcome , Gradio Deployment in huggingface_hub'
     '</div>')



Support Vector Classifier for categorical outcome , Gradio Deployment in huggingface_hub



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # Support Vector Classifier for categorical outcome
from sklearn.preprocessing import StandardScaler # Essential for SVC
import gradio as gr
import joblib
import os

# --- 1. Data Loading and Preprocessing (Replicated for standalone app.py) ---
# This section ensures the app.py can run independently by performing all data prep.

# Load the dataset from the provided GitHub URL
try:
    df = pd.read_csv('https://raw.githubusercontent.com/rick7122003/Performance-Predictive-Analytics/main/salesforce%20data.csv')
except Exception as e:
    print(f"Error loading data from URL: {e}. Creating a dummy DataFrame for demonstration.")
    # Create a dummy DataFrame for demonstration if file loading from URL fails
    data = {
        'Account ID': ['A1', 'A2'], 'Account Name': ['Comp A', 'Comp B'],
        'Account Owned By': ['John', 'Jane'], 'Account Type': ['Gold', 'Silver'],
        'Address': ['123 Main', '456 Oak'], 'Billing City': ['NY', 'LA'],
        'Billing Region': ['East', 'West'], 'Billing State': ['NY', 'CA'],
        'Close Date': ['1/1/2023', '2/1/2023'], 'Closed': [True, True],
        'Created Date': ['1/1/2023', '1/15/2023'], 'Current Quarter?': ['Yes', 'No'],
        'Days Left before EOQ': [30, 60], 'Hide Dates': [False, False],
        'Industry': ['Tech', 'Finance'], 'Opportunity ID': ['OP1', 'OP2'],
        'Opportunity Name': ['Deal 1', 'Deal 2'], 'Opportunity Type': ['Software', 'Hardware'],
        'Opportunity Type Filter': [0, 0], 'Product Name': ['ProdX', 'ProdY'],
        'Product Name Filter': [0, 0], 'Stage': ['Closed Won', 'Closed Lost'],
        'Today': ['3/1/2023', '3/1/2023'], 'Won': [True, False], 'Zip Code': [10001, 90210],
        'Amount': [50000.0, 10000.0], 'Customer Count': [1, 1], 'Discount Granted': ['10.0%', '5.0%'],
        'Latitude': [40.0, 34.0], 'Longitude': [-74.0, -118.0], 'Number of Records': [1, 1],
        'Opportunity Quantity': [50.0, 10.0], 'Sales': ['$50K', '$0K'], 'Software Sold': [50, 0],
        'Transactions': [1, 0], 'Unit Price': [1000, 1000], 'Weekly Sales': ['$1000', '$0'],
        'Current Quarter Quota': ['$1M', '$0M'], 'Quota': [100000.0, 50000.0]
    }
    df = pd.DataFrame(data)


# Drop redundant columns (columns with '.1' suffix)
df = df.drop(columns=[col for col in df.columns if col.endswith('.1')], axis=1, errors='ignore')

# Convert date columns to datetime objects
date_cols = ['Close Date', 'Created Date', 'Today']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Clean and convert monetary and percentage columns to numeric
df['Sales'] = df['Sales'].replace({r'[$,]': '', 'K': 'e3', 'M': 'e6'}, regex=True).astype(float)
df['Weekly Sales'] = df['Weekly Sales'].replace({r'[$,]': '', ',': ''}, regex=True).astype(float)
df['Current Quarter Quota'] = df['Current Quarter Quota'].replace({r'[$,]': '', 'M': 'e6'}, regex=True).astype(float)
df['Discount Granted'] = df['Discount Granted'].str.replace('%', '', regex=False).astype(float) / 100

# Convert boolean columns to integers (0 or 1)
df['Closed'] = df['Closed'].astype(int)
df['Won'] = df['Won'].astype(int)

# Categorize 'Sales' into 'Low Sales' and 'High Sales' based on median
median_sales = df['Sales'].median()
df['Sales_Category'] = df['Sales'].apply(lambda x: 'High Sales' if x > median_sales else 'Low Sales')

# Define target and features with the expanded list
target_category = 'Sales_Category'
features_for_classification = [
    'Industry', 'Product Name', 'Opportunity Type', 'Amount', 'Opportunity Quantity',
    'Discount Granted', 'Customer Count', 'Account Type', 'Billing Region', 'Stage'
]

# Filter DataFrame to include only the relevant columns for this analysis
df_model_prep = df[features_for_classification + [target_category]].copy()

# Identify numerical and categorical features for imputation and scaling/encoding
numerical_cols_for_imputation = df_model_prep[features_for_classification].select_dtypes(include=np.number).columns
categorical_cols_for_imputation = df_model_prep[features_for_classification].select_dtypes(include='object').columns

# Impute numerical missing values with the median
for col in numerical_cols_for_imputation:
    if df_model_prep[col].isnull().sum() > 0:
        df_model_prep[col] = df_model_prep[col].fillna(df_model_prep[col].median())

# Impute categorical missing values with the mode
for col in categorical_cols_for_imputation:
    if df_model_prep[col].isnull().sum() > 0:
        df_model_prep[col] = df_model_prep[col].fillna(df_model_prep[col].mode()[0])

# Drop rows where the target variable itself might be missing
df_model_prep.dropna(subset=[target_category], inplace=True)


# Separate features (X) and target (y)
X = df_model_prep[features_for_classification]
y = df_model_prep[target_category]

# Apply One-Hot Encoding to categorical features
X_encoded = pd.get_dummies(X, columns=categorical_cols_for_imputation, drop_first=True)

# Apply Feature Scaling to numerical features after one-hot encoding
numerical_cols_after_encoding = [col for col in X_encoded.columns if col in numerical_cols_for_imputation]

scaler = StandardScaler()
X_scaled_encoded = X_encoded.copy() # Create a copy to avoid SettingWithCopyWarning
X_scaled_encoded[numerical_cols_after_encoding] = scaler.fit_transform(X_encoded[numerical_cols_after_encoding])


# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize and train the Support Vector Classifier (SVC) model
model_svc = SVC(kernel='rbf', random_state=42, probability=True)
model_svc.fit(X_train, y_train)

# --- 2. Gradio Prediction Function ---

# Store the columns used during training for consistent input processing
training_columns = X_train.columns.tolist()

def predict_sales_category_svc(
    industry, product_name, opportunity_type, amount, opportunity_quantity,
    discount_granted, customer_count, account_type, billing_region, stage
):
    # Create a DataFrame from the input values
    input_data = pd.DataFrame([[
        industry, product_name, opportunity_type, amount, opportunity_quantity,
        discount_granted, customer_count, account_type, billing_region, stage
    ]], columns=features_for_classification)

    # Ensure correct data types for numerical columns in input_data
    for col in numerical_cols_for_imputation:
        input_data[col] = pd.to_numeric(input_data[col])

    # Apply One-Hot Encoding to the input data (must be consistent with training)
    input_encoded = pd.get_dummies(input_data, columns=categorical_cols_for_imputation, drop_first=True)

    # Align columns: Add missing columns (from training set) and remove extra ones
    missing_cols = set(training_columns) - set(input_encoded.columns)
    for c in missing_cols:
        input_encoded[c] = 0
    input_encoded = input_encoded[training_columns] # Ensure column order is same as training

    # Apply Feature Scaling to numerical features in input data (must be consistent with training)
    numerical_cols_in_training = [col for col in training_columns if col in numerical_cols_for_imputation]
    
    input_scaled_encoded = input_encoded.copy()
    input_scaled_encoded[numerical_cols_in_training] = scaler.transform(input_encoded[numerical_cols_in_training])

    # Make prediction
    prediction = model_svc.predict(input_scaled_encoded)[0]

    return f"Predicted Sales Category: {prediction}"

# --- 3. Gradio Interface Setup ---

# Get unique values for dropdowns from the preprocessed DataFrame
unique_industries = df_model_prep['Industry'].unique().tolist()
unique_product_names = df_model_prep['Product Name'].unique().tolist()
unique_opportunity_types = df_model_prep['Opportunity Type'].unique().tolist()
unique_account_types = df_model_prep['Account Type'].unique().tolist()
unique_billing_regions = df_model_prep['Billing Region'].unique().tolist()
unique_stages = df_model_prep['Stage'].unique().tolist()

# Define Gradio input components
inputs = [
    gr.Dropdown(unique_industries, label="Industry", value=unique_industries[0] if unique_industries else None),
    gr.Dropdown(unique_product_names, label="Product Name", value=unique_product_names[0] if unique_product_names else None),
    gr.Dropdown(unique_opportunity_types, label="Opportunity Type", value=unique_opportunity_types[0] if unique_opportunity_types else None),
    gr.Number(label="Amount ($)", value=10000.0), # Example default value
    gr.Number(label="Opportunity Quantity", value=10.0),
    gr.Number(label="Discount Granted (e.g., 0.15 for 15%)", value=0.10),
    gr.Number(label="Customer Count", value=1.0),
    gr.Dropdown(unique_account_types, label="Account Type", value=unique_account_types[0] if unique_account_types else None),
    gr.Dropdown(unique_billing_regions, label="Billing Region", value=unique_billing_regions[0] if unique_billing_regions else None),
    gr.Dropdown(unique_stages, label="Stage", value=unique_stages[0] if unique_stages else None)
]

# Define Gradio output component
output = gr.Textbox(label="Prediction Result")

# Create the Gradio Interface
iface = gr.Interface(
    fn=predict_sales_category_svc,
    inputs=inputs,
    outputs=output,
    title="Sales Category Predictor (SVC Model)",
    description="Predict if a sales opportunity will result in 'High Sales' or 'Low Sales' based on its characteristics. "
                "**Note: The model's high accuracy is largely due to the 'Amount' feature being a strong indicator of 'Sales Category'.**"
)

# Launch the Gradio app
iface.launch()


<img width="708" height="381" alt="image" src="https://github.com/user-attachments/assets/93a081ba-137e-42fc-81bf-d67fa13b4fd7" />




<img width="731" height="388" alt="image" src="https://github.com/user-attachments/assets/5817a1a4-7e77-40d1-b36a-f35464e94df9" />




<img width="739" height="394" alt="image" src="https://github.com/user-attachments/assets/850d7e85-3622-4bcb-ac1b-c648e6e415c8" />



Success Sales Predictable ML model is deployed !









