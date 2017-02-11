# Loan Data - Binary Classification
Final project for the Microsoft Professional Program for Data Science

(I wish that I had thought of tracking each step on GitHub but I am doing this after the fact so I only have my final and best model.  I'll do better about tracking iterations of my Data Science projects in the future.  Also note that my inital models scored in the 40s because I was WAAAAY overfitting my model due to cleaning that was too specific and didn't generalize well to the testing data.)

The following are details of the machine learning model that was subsequently published as a Predictive Web Service using Microsoft's Azure Machine Learning Studio.  It is the Final Project of the  Microsoft Professional Program for Data Science and it takes the form of a Cortana Intelligence Competition. The goal of the competition is to predict (as a binary classification problem), whether a loan will be fully repaid with a sufficient enough score to pass the course which in this case is greater than 70% accuracy.  

We are given a Public Training Data Set.  When we submit our model to the Cortana Platform, a Public Test Data Set is run through the Predictive Model and returns an accuracy score.  Finally, once the deadline has passed, we give the Submission ID for our most successful (highest scoring) model and a Private Test Data Set is run through and an accuracy score returned.  If that final accuracy score is greater than 70%, I pass the course and receive my Certificate of Data Science and will be a Microsoft Certified Professional (MCP)!

# Public Training Data Set
Upon downloading this data set, I opened it up in Excel and explored it a bit just to get an inital feel for the features it contained.  I then loaded it into R Studio and did some further exploration.  Lastly, I uploaded it to Azure ML and noted that the following had been done to the data set to 'muck it up' in order to test our exploration and cleaning skills (values are from Azure ML):

|Feature Name | Unique Values | Missing Values | Min/Max | Type | Notes |
|---------|------|---|---|---|---|--------------|
|Loan ID | 88,910 | 0 | | String |                                  |
|Customer ID | 88,910 | 0 | | String |                              |
|Loan Status | 2 | 0 | | String | This is the label |
|Current Loan Amount | 22,541 | 0 | 491/99999999 | Numeric | (12,738 rows with 99999999) |
|Term | 2 | 0 | | String | |
|Credit Score | 167 | 21,338 | 585/7510 | Numeric | Score should not be higher than 800 |
|Years in Current Job | 12 | 0 | | String | 3,817 n/a values |
|Home Ownership | 4 | 0 | | String | |
|Annual Income | 37,983 | 21,338 | 4,033/8,713,547 | Numeric | |
|Purpose | 16 | 0 | | String | |
|Monthly Debt | 70,067 | 0 | 0/22,939.12 | String | $ values |
|Years of Credit History | 508 | 0 | 3.6/70.5 | Numeric | |
|Months since Last Delinquent | 117 | 0 | | String | NA values |
|Number of Open Accounts | 52 | 0 | | Numeric | |
|Number of Credit Problems | 14 | 0 | | Numeric | |
|Current Credit Balance | 33,716 | 0 | 0/1,730,472 | Numeric | |
|Maximum Open Credit | 46,625 | 0 | 0/22,939.12 | String | 2 rows with #Value |
|Bankrupties | 9 | 0 | | String | NA values |
|Tax Liens | 14 | 0 | | String | NA values |
|Number of Starting Records | 111,107 | 19 columns| | | |

# Set up Indicator Columns
I noticed that if the 'Current Loan Amount' is set to 99999999 that the corresponding 'Loan Status' is Fully Paid.  Seems like the amount really should be 0, so I'm going to add a column that indicates these rows with a 1 and a 0 otherwise.  Then, I'm going to set 99999999 to NA (I'll convert it to 0 later).  

Then I decided to add another indicator column for those times that 'Credit Score' has a trailing zero, making the credit score 4 digits instead of 3 which obviously makes those instances greater than the highest credit score possible of 850.

Next, add an indicator for missing 'Annual Income.'

Then, once all desired indicator columns are added to create a 'path' for the machine learning algorithm to follow, divide 4 digit credit scores by 10 and filter out missing rows.  

* Number of rows remaining 89,769 and 22 columns

```
library("dplyr")
library("magrittr")
library("tidyr")

df <- df %>%
mutate("Current Loan Amount High" = ifelse(df$"Current Loan Amount" == 99999999, 1, 0))%>%
mutate("Current Loan Amount" = ifelse(df$"Current Loan Amount" == 99999999, NA, df$"Current Loan Amount"))%>%
mutate("Credit Score High" = ifelse(df$"Credit Score" > 850, 1, 0))%>%
mutate("Annual Income Blank" = ifelse(is.na(df$"Annual Income"), 1, 0)) %>%
mutate("Credit Score" = as.numeric(ifelse(df$"Credit Score" > 850, df$"Credit Score"/10, df$"Credit Score"))) %>%
filter(!is.na(df$"Credit Score"))
```
# Remove extra characters that are making what should be Numeric columns become String type

* Note: For some reason I was having weird issues in Azure ML if I created one large custom R script, so I separated my cleaning process out into several sections to get it to work

Here's the next bit of R code:
This removes the $ sign from 'Monthly Debt' and then converts it into Numeric type.  Then, it combines 2 'Purposes' that appear the same (taking it from 16 to 15 possibilities), and 2 'Home Ownership' types that mean the same thing (taking it from 4 to 3 possibilities) 
```
library(dplyr)
library(magrittr)
library(tidyr)

df <- df %>%
mutate("Monthly Debt" = stringr::str_replace(df$"Monthly Debt","\\$","")) %>% 
mutate("Monthly Debt" = as.numeric(df$"Monthly Debt")) %>%
mutate("Purpose" = ifelse(Purpose %in% c('Other', 'other'),
                                'Other', Purpose)) %>%
mutate("Home Ownership" = ifelse(df$"Home Ownership" == "HaveMortgage", "Home Mortgage",df$"Home Ownership"))
```
# Remove duplicates
We need to remove duplicates so that our results are not skewed.  The following R script orders the data frame by 'Loan ID,' then by 'Credit Score' in descending order and then by 'Current Loan Amount.'  This makes it so that the rows that have complete data are retained over those with missing data.

* Note:  This R script code block is to be deleted in the Predictive Experiment so that when the Public Test Data and Private Test Data are run through the model, rows are not unnecessarily deleted causing a 0 score for that row and resulting in a lower accuracy score.

* Number of rows remaining is 72,821, exactly the number of unique Loan IDs at this point.

```
df <- df[order(df$"Loan ID", -df$"Credit Score", df$"Current Loan Amount"),] #111107 rows
df <- df[!duplicated(df$"Loan ID"),]
```
# More combining, add ratios commonly used in the loan industry, first step to converting 'Years in current job' to numeric, and make a few data transformations
The following R code combines 2 more categories of 'Purpose' that appear the same, taking it from 15 possibilities to 14.

Then we add the following ratios:  Debt to Income (DTI), Loan to Income, and Debt to Loan.

Looking at the hisograms generated in Azure ML at just prior to this point, it appeared that the following features might benefit from a log transformation: 'Annual Income,' 'Current Loan Amount,' and 'Maximum Open Credit'

Next, this script uses the R package 'stringr' to remove the word 'years' from 'Years in current job.'

Finally, convert 'Bankruptcies' and 'Tax Liens' to Numeric which were previously String type due to NA values.  

```
library(dplyr)
library(magrittr)
library(tidyr)

df <- df %>%
mutate("Purpose" = ifelse(Purpose %in% c('Take a Trip', 'vacation'),
                          'Vacation', Purpose)) %>%
mutate("DTI" = df$"Monthly Debt"/df$"Annual Income"/12*100) %>%
mutate("Loan To Income Ratio" = df$"Current Loan Amount"/df$"Annual Income") %>%
mutate("Debt To Loan Ratio" = log(df$"Monthly Debt"/df$"Current Loan Amount")) %>%
mutate("Annual Income" = log(df$"Annual Income")) %>%
mutate("Current Loan Amount" = log(df$"Current Loan Amount")) %>%
mutate("Maximum Open Credit" = log(as.numeric(df$"Maximum Open Credit"))) %>%
mutate("Current Credit Balance"= log(df$"Current Credit Balance")) %>%
mutate("Years in current job" = stringr::str_replace(df$"Years in current job", " years*","")) %>%
mutate("Bankruptcies" = as.numeric(df$"Bankruptcies")) %>%
mutate("Tax Liens" = as.numeric(df$"Tax Liens"))  
```
# More combining, next step to converting 'Years in current job' to numeric, remove infinity from log transforms
Last combining step of 2 categories of 'Purpose' that have the same meaning taking it down to 13 possibilities (originally 16).

Remove the '+' symbol from 'Years in current job' as a next step in transforming this feature from String to Numeric.

Finally, remove those pesky infinity that happened during the log transformation of 'Debt To Loan Ratio,' 'Maximum Open Credit,' and 'Current Credit Balance' in previous script.

```
library(dplyr)
library(magrittr)
library(tidyr)

df <- df %>%
mutate("Purpose" = ifelse(Purpose %in% c('small_business','Business Loan'),
                          'Business Loan', Purpose)) %>%
mutate("Years in current job" = stringr::str_replace(df$"Years in current job", "\\+", "")) %>%
mutate("Debt To Loan Ratio" = ifelse(is.infinite(df$"Debt To Loan Ratio"), NA, df$"Debt To Loan Ratio")) %>%
mutate("Maximum Open Credit" = ifelse(is.infinite(df$"Maximum Open Credit"), NA, df$"Maximum Open Credit")) %>%
mutate("Current Credit Balance"= ifelse(is.infinite(df$"Current Credit Balance"), NA, df$"Current Credit Balance"))
```

# Replace NaNs, last step to converting 'Years in current job' to Numeric
Need to address the NaN weirdness that happened during the log transformation of 'Debt To Loan Ratio,' 'Maximum Open Credit,' and 'Current Credit Balance' in a previous script by replacing with NA instead.

Then, replace the '< 1' with 0 in 'Years in current job' and convert to Numeric.

```
library(dplyr)
library(magrittr)
library(tidyr)

df <- df %>%
mutate("Debt To Loan Ratio" = ifelse(is.nan(df$"Debt To Loan Ratio"), NA, df$"Debt To Loan Ratio")) %>%
mutate("Maximum Open Credit" = ifelse(is.nan(df$"Maximum Open Credit"), NA, df$"Maximum Open Credit")) %>%
mutate("Current Credit Balance" = ifelse(is.nan(df$"Current Credit Balance"), NA, df$"Current Credit Balance")) %>%
mutate("Years in current job" =  stringr::str_replace(df$"Years in current job", "< 1", 0)) %>%
mutate("Years in current job" = as.numeric(df$"Years in current job"))
```

# Clean Missing Data
The next step (finally) is to fill in missing data.  This required 3 separate Clean Missing Data modules in Azure ML.  

The first is for 'Current Loan Amount,' 'Credit Score,' 'Annual Income,' 'Monthly Debt,' 'Current Credit Balance,' 'Maximum Open Credit,' 'DTI,' 'Loan To Income Ratio,' and 'Debt To Loan Ratio.'  I chose to fill in missing values with the median of the feature.

The second is for 'Years in current job,' 'Bankruptcies,' 'Tax Liens,' 'Current Loan Amount High,' and 'Credit Score High.'  I chose to fill in those missing values with 0.

The third is for 'Months since last delinquent' which I chose to fill with 177.  Why such a weird number?  Well because all those NAs were most likely caused by that particular loan recipient not having a late payment for a very long time. So I just took the max number of months and added 1.  Voila - 177!  It would not have made sense to use 0, because that would instead indicate that they were late in the last month which is much less likely.

# Edit Metadata
The next step was to use an Edit Metadata module to select Loan ID and Customer ID and choose to 'Clear Feature.'  This way, these features are not considered in the model.

# Another Edit Metadata
I then used another Edit Metadata module, but this time as a way to specify to the model that 'Loan Status' is to be used as our label.

# Split Data
Time to split the data using a Split Data module.  I used the Split Rows splitting mode, 0.7 for my training split, randomized split checked, random seed set to 1234, and Stratified Split set to FALSE.

# Two-Class Boosted Decision Tree
I used the Two-Class Boosted Decision Tree machine learning algorithm to feed into my training split of the data set.  I used a Tune Hyperparameters module to tune the parameters of the Decision Tree (DT) as well as a Cross Validate module in order to make sure there was not a significant change between folds and that the standard deviation was several orders of magnitude below the mean of each metric which indicates the model will generalize well in production.  I used the following parameters for the DT:

|Create trainer mode | Maximum number of leaves per tree | Minimum number of samples per leaf node | Learning rate | Number of trees constructed | Random number seed | Allow unknown categorical levels |
|-----------|------|---|---|---|---|--------------|
|Single parameter | 28 | 39 | 0.030421 | 429 | 1234 | Checked |

# Train, Score, Evaluate
Finally, time to train, then score and evaluate this model.  

Accuracy | Public Training Data Set| Public Testing Data Set | Private Testing Data Set |
|---|---|---|---|
|  | .785 | .7243 | .7251 |

Yay!  I achieved a passing score with this model, making me eligible for my Certification for the Microsoft Professional Program in Data Science.