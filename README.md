# Recognizing-hand-written-digits
Using the data set of the numbers included in the learn-scikit library:
http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits
Using existing code from :
https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

We built features to classify the data set.
The features we chose:
The features we chose:
1. The sum of the values on the main diagonal.
2. Symmetry of the matrix which is calculated by rotating the matrix by 180 degrees, subtracting from the original an absolute value and the sum of all the remaining values.
3. Calculation of the longest vertical line (consecutive painted pixels that are not 0 (white).
4. Calculating the average value of the color in the 4 central pixels.
5. Calculation of the average number of points of intersection of a horizontal line in each row with the number (we calculate this by cutting a sequence of 0).

Display quality classified by graphs
