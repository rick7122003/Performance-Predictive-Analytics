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






















