# Regression-Project-Zillow
### Description 
- Project examining drivers of single unit home assessed tax value in California during May-Aug 2017 using the zillow database from Codeup

### Goals
- The goals are to find drivers of assessed tax value and to predict it using a machine learning model
- Include state, county names, and tax rate distribution for each county.
---------------------------------
### Data Dictionary
---
| Column | Definition | Data Type |
| ----- | ----- | ----- |
|bathrooms | number of bathrooms in home including half baths| float|
|bedrooms| number of bedrooms in home| float|
|square_feet| total finished living area of home| float|
|fips| federal county code| float|
|latitude| latitude of the middle of the parcel | float|
|longitude| longitude of the middle of the parcel| float|
|lotsizesquarefeet| area of the lot in sqft| float|
|regionidzip| zip code the property is located in| float|
|taxes| taxes paid for assement year| float|
|transactiondate| date the property sold| object|
|age_in_years| houses age from yearbuilt to 2021| float|
|Bathrooms_cat| categorical bathrooms from 0,1,2,3,4+|object|
|Bedrooms_cat| categorical bedrooms from 0,1,2,3,4+| object|
|tax_rate| taxes paid divided by assessed tax value perctenage| float|

---------------------------------------------------
| Target | Definition | Data Type |
| ----- | ----- | ----- |
|tax_value|total taxed assessed value of the parcel| float |

### Hypotheses
**1. Is the mean tax value significantly different between different counties?**
- null_hypothesis = "The mean tax values are the same for between counties"
- alternative_hypothesis = "The mean tax values are significantly different between counties"

**2. Is there a correlation between home age and tax_value?**
- null_hypothesis = "There is no correlation between home age and tax_value"
- alternative_hypothesis = "There is a negative correlation between home age and tax value"

**3. Is there a correlation between tax value and square footage?**
- null_hypothesis = "There is no correlation between tax value and square footage"
- alternative_hypothesis = "There is a positive correlation between tax value and square footage"

### Project Plan
1. Acquire data from zillow database using correct joins
2. Prepare data by removing unnecessary/redundant columns, dealing with nulls, and encoding variables for use with ML models
3. Explore data using functions in explore.py as well as jointplot, pairplot, and heatmap.
4. Create ML models and choose best performing for test data
5. Present findings on tax value drivers, tax rate distributions between counties, and regression model performance

### How to re-create
- All necessary files minus env.py are in this repository so the best method would be to git clone and add your own env.py if you want to create your own zillow.csv
- Run Regression_Project_Zillow.ipynb
- Adjust exploration and modeling to your liking
