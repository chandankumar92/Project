ELECTRICITY USAGE PREDICTOR
PS No. 71 | Rungta International Skills University
B.Tech Computer Science and Engineering

==================================================
DESCRIPTION
==================================================
A Python program that predicts future electricity usage from past
monthly data. It loads a CSV file with Pandas, fits a simple linear
regression line with NumPy, predicts the units for a future month,
estimates the electricity bill and plots the actual data against the
trend line with Matplotlib.

==================================================
FEATURES
==================================================
- Loads monthly usage from a CSV file and checks for missing or
  non-numeric values
- Shows a short data summary (rows, months, min, max and mean units)
- Fits a regression line using numpy.polyfit (degree 1):
  Units = slope x Month + intercept
- Calculates the R2 value (rounded to 2 decimals)
- Predicts units for any month entered by the user
- Estimates the bill: predicted units x Rs. 6 per unit
- Shows an error if fewer than 2 months of data are available
- Plots actual usage (points) and the regression line with a title,
  labels and legend
- Saves the chart as usage_plot.png
- Simple menu: Predict, Exit, and Run all 4 test cases

==================================================
TECHNOLOGIES USED
==================================================
- Python 3
- Pandas
- NumPy
- Matplotlib

==================================================
CSV FORMAT
==================================================
Each line is: Month,Units
The file must be named usage.csv and kept in the same folder as the
script.

Example:
1,200
2,220
3,240
4,260

==================================================
HOW IT WORKS
==================================================
1. Pandas loads the CSV file and checks the data for missing or
   non-numeric values.
2. NumPy fits the regression line and calculates the slope,
   intercept and R2 value.
3. The predicted units for the entered month are calculated and
   multiplied by Rs. 6 to estimate the bill.
4. Matplotlib plots the actual usage and the trend line.
