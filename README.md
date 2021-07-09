# RGB to LAB converter
Dylan Pilz

Last updated: 8.7.2021

#Overview
The CIELAB colorspace is of great importance in food image evaluation
since it mirrors human color perception more accurately than the typical
RGB colorspace used by most cameras. It is therefore better suited for
predicting a consumer's perception of food quality.

In this project, I used several approaches to attempt to recreate the results of
León et al, which seeks an accurate approach to convert pixel values from RGB to LAB
for use in food evaluation.

The data is taken from an image of a standard color checker, with known L*a*b* values.

#Linear regression (lin_regress.py):

Linear regression approach, using linear_model from sklearn library.

subset_data(rgb_file, LAB_file)
Clean input .txt files and return them as numpy arrays.

plot_regression(X, y, title)
Produce scatter plot and best fit line

calculate_stats(expected, predicted, title)
Prints summary statistics(shown below) and returns mean absolute error


L value - linear model
-------------------------------------
Mean abs error: 4.5966228965582285
Median error: 3.8848823718920507
Min error: 0.3953531196788731 at index 3
Max error: 13.433412731378581 at index 18
-------------------------------------

a value - linear model
-------------------------------------
Mean abs error: 7.017837540294636
Median error: 4.722660574701337
Min error: 0.3726550914651559 at index 8
Max error: 20.385482594892615 at index 24
-------------------------------------

b value - linear model
-------------------------------------
Mean abs error: 4.7359271862145285
Median error: 2.5329451499982945
Min error: 0.1663003888353387 at index 17
Max error: 25.092210990241433 at index 11
-------------------------------------


#Quadratic regression(quad_regress.py):

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

L value - quadratic model
-------------------------------------
Mean abs error: 2.187636465294087
Median error: 2.1823326437838837
Min error: 0.3374057248608935 at index 13
Max error: 5.538765782499933 at index 18
-------------------------------------

a value - quadratic model
-------------------------------------
Mean abs error: 3.0173261809247562
Median error: 1.303165311489248
Min error: 0.06980233171595751 at index 37
Max error: 11.985517986656236 at index 1
-------------------------------------

b value - quadratic model
-------------------------------------
Mean abs error: 2.4048787362640494
Median error: 1.343226117898225
Min error: 0.016768196851240147 at index 31
Max error: 9.724663685571048 at index 22
-------------------------------------


#Neural network(neural_net.py):

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

L value NN
-------------------------------------
Mean abs error: 2.649895610809326
Median error: 1.9026937
Min error: [0.01737976] at index 39
Max error: [9.156391] at index 23
-------------------------------------
D

a value NN
-------------------------------------
Mean abs error: 6.881444798878262
Median error: 5.9735413
Min error: [0.13208336] at index 18
Max error: [24.364315] at index 25
-------------------------------------


b value NN
-------------------------------------
Mean abs error: 5.640956619367712
Median error: 3.8693168
Min error: [0.00230414] at index 16
Max error: [18.158203] at index 2
-------------------------------------


Summary table: 

-----------  -----------------  ------------------  --------------------
Value\Model  Neural network     Linear regression   Quadratic regression
L*           2.649895610809326  4.5966228965582285  2.187636465294087
a*           6.881444798878262  7.017837540294636   3.0173261809247562
b*           5.640956619367712  4.7359271862145285  2.4048787362640494
-----------  -----------------  ------------------  --------------------


References:

Katherine León, Domingo Mery, Franco Pedreschi, Jorge León,
Color measurement in L∗a∗b∗ units from RGB digital images,
Food Research International,
Volume 39, Issue 10,
2006,
Pages 1084-1091,
ISSN 0963-9969,
https://doi.org/10.1016/j.foodres.2006.03.006.
