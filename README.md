# RGB to LAB converter
Dylan Pilz

Last updated: 7/9/2021

# Overview
The CIELAB colorspace is of great importance in food image evaluation
since it mirrors human color perception more accurately than the typical
RGB colorspace used by most cameras. It is therefore better suited for
predicting a consumer's perception of food quality.

In this project, I used several approaches to attempt to recreate the results of
León et al, which seeks an accurate approach to convert pixel values from RGB to LAB
for use in food evaluation.

The data is taken from an image of a standard color checker, with known L*a*b* values.

# Linear regression (lin_regress.py):

Linear regression approach, using linear_model from sklearn library.

subset_data(rgb_file, LAB_file)
Clean input .txt files and return them as numpy arrays.

plot_regression(X, y, title)
Produce scatter plot and best fit line

calculate_stats(expected, predicted, title)
Prints summary statistics(shown below) and returns mean absolute error

# Quadratic regression(quad_regress.py):

Quadratic regression, again using linear_model but with RG, RB, GB, R^2, G^2, B^2 appended to input array

subset_data(rgb_file, LAB_file)
Clean input .txt files and return them as numpy arrays.

quad_setup(rgb_data)
Additional preprocessing step, appending RG, RB, GB, R^2, G^2, B^2 to input array

plot_regression(X, y, title)
Produce scatter plot and best fit line

calculate_stats(expected, predicted, title)
Provides summary statistics(shown below) and returns mean absolute error

quad_setup(rgb_data)


# Neural network(neural_net.py):

Splits LAB input into seperate columns, and trains a seperate model for each, with 2 hidden layers containing
4 nodes each. Performs decently well, but could be improved significantly given more training data. The accuracy
for L values tapers off at the extremes, but this shouldn't be too much of an issue when evaluating meat since
the lightness values remain fairly centered around 50.

subset_data(rgb_file, LAB_file, col)
Parse input files and output numpy array, the col parameter specifies which column (L, a, or b) to select from the
LAB_file. Scales L* values down by a factor of 100, a* and b* values by 128

build_model(output_activation)
Construct a model with the provided output activation function (sigmoid for L*, tanh for a* and b*)
Contains 2 hidden dense layers with 4 nodes each.

plot_loss(hist, label)
plots the output of the loss function for each epoch


# Summary table: 
Mean absolute errors for the three models are shown below:
| Value\Model | Neural network     | Linear regression  | Quadratic regression |
|-------------|--------------------|--------------------|----------------------|
| L*          | 1.8639297685169036 | 4.5966228965582285 | 2.187636465294087    |
| a*          | 6.498623516630558  | 7.017837540294636  | 3.0173261809247562   |
| b*          | 2.848936311006546  | 4.7359271862145285 | 2.4048787362640494   |


# References:
Katherine León, Domingo Mery, Franco Pedreschi, Jorge León,
Color measurement in L∗a∗b∗ units from RGB digital images,
Food Research International,
Volume 39, Issue 10,
2006,
Pages 1084-1091,
ISSN 0963-9969,
https://doi.org/10.1016/j.foodres.2006.03.006.
