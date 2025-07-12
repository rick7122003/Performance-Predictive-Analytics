# Capstone AI & Machine Learning: Sales prediction

Sales Performance prediction






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



