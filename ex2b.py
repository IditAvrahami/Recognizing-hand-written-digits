"""
================================
Recognizing hand-written digits
================================

Names: Valeri Materman 321133324
       Idit Avrahami 207565748

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np


# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics,preprocessing,linear_model,model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, metrics
from sklearn import linear_model
from sklearn. model_selection import cross_val_predict
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier





def getWrongClassificationsFromDS(X_test,predicted,y_test):
    '''
    Get the wrong classified images using the classifier from the sklearn website.
    Parameters
    ----------
    X_test : Dataset of images [8*8]
        Dataest that user ran the test on
    predicted : 0-9 numbers array
        The prediction of the classifer
    y_test : 0-9 numbers array
        The real value of each image

    Returns
    -------
    None.

    '''

    number_of_diff = np.count_nonzero(predicted!=y_test)
    n_rows = number_of_diff//10

    figure = plt.figure(figsize=(11,6),facecolor=("grey"))

    figure.suptitle('Test. miss-classification: exptected - predicted', fontsize=16)

    i = 0
    
    for image, prediction,true_class in zip(X_test, predicted,y_test):
 
        if prediction == true_class:
            continue
        
        image = image.reshape(8, 8)
        ax = plt.subplot(n_rows+1,10,i+1)
        ax.set_axis_off()    
        ax.set_title(f"{true_class} {prediction}")
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")       
        i= i+1
        
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
      
    plt.show()
    
def extractSpecificIndexDigitsFromDS(digit_1, digit_2,ds):
    '''
    extracting specific indecies from the data set based  on the values of the numbers
    in the range >= digits_1 and <= digits_2

    Parameters
    ----------
    digit_1 : int
        The first number.
    digit_2 : int
        The secodn number.
    ds : TYPE
        The data set itself.

    Returns
    -------
    Indecies of the members in that value range as np array.

    '''

    return np.where(np.logical_and(digits.target >=digit_1 , digits.target <= digit_2))

#######################################################################################################################
#Feature functions:

def longestVerticalLine(image):
    '''
    Longest vertical line that has color (non zero value) in the image

    Parameters
    ----------
    image : [8*8] array
        the image

    Returns
    -------
    max_non_zero_count : double
        The length of the longest vertical line that has color.

    '''
    rot_m = np.rot90(image,1)
    curr_non_zero_count =0;
    max_non_zero_count= 0;
    
    for row in rot_m:
        for value in row:
            if value == 0:
                if curr_non_zero_count > max_non_zero_count:
                    max_non_zero_count = curr_non_zero_count
                curr_non_zero_count = 0
            else:
                curr_non_zero_count+=1
                
    return max_non_zero_count

def averageOfVerticalMiddleLine(image):
    '''
    The average value of the verical middle line

    Parameters
    ----------
    image : [8*8] array
        the image

    Returns
    -------
    double
        average of vertical middle line.
    '''
    rot_m = np.rot90(image,1)
    return np.average(rot_m[3])

def simetry(image):
    '''
    Checks either the function is semetrical based on the sum of a subtraction between the 
    matrix and the matrix rotated 180 deg.

    Parameters
    ----------
    image : [8*8] array
        the image

    Returns
    -------
    double
        the sum as explained above.

    '''
    rot_m = np.rot90(image,2)
    return np.sum(abs(image - rot_m))

    
def sumOfValuesOnDiagonal(image):
    '''
    Calculates the average of the values on the main diagonal

    Parameters
    ----------
    image : [8*8] array
        the image

    Returns
    -------
    double
        as expalined above.

    '''
    return np.sum(np.diagonal(image))

def averageColorOfMiddle(image):
    '''
    Average color in the middle 4 pixels

    Parameters
    ----------
    image : [8*8] array
        the image

    Returns
    -------
    double
        average as explained above.

    '''
    return np.average(image[np.array([3,3,4,4]),np.array([3,4,3,4])])

def averageRowIntersection(image):
    """
    Calculates the average intersection in any row
    intersection will be calculated as how many times we will meet colored pixels
    after seeing white pixels.

    Parameters
    ----------
    image : [8*8] array
        the image

    Returns
    -------
    double
        the average as explained above.

    """
    inter = np.array([])
    n_of_intersections= 0
    color = False
    
    for row in image:
        for value in row:
            if value == 0 and color == True:
                n_of_intersections +=1
                color = False
            else:
                if value == 0:
                    continue
                color = True
        inter = np.append(inter,n_of_intersections)
        n_of_intersections =0

        
    return np.average(inter)


def varMatrix(image):
    """
    calculates the variance of the matrix

    Parameters
    ----------
    image : [8*8] array
        the image

    Returns
    -------
    double
        the varienc of the imag.

    """
    return np.var(image) 

def number_of_hori_lines(image):
    """
    Calculates the number of horizontal lines buy searchign lines that are larger then
    2 pixels and are not fully black
    ----------
    image : [8*8] array
        the image

    Returns
    -------
    double
        the number of such lines

    """
    image = np.rot90(image,1)
    n_of_lines = 0;
    color_p = False
    number_of_b = 0
    for row in image:
        color_p = False
        for value in row:
            if value <16 and color_p == True:
                if number_of_b >2:
                    number_of_b = 0
                    n_of_lines = n_of_lines +1
                else:
                    color_p = False
            else:
                number_of_b = number_of_b +1
                color_p = True
        if number_of_b != 0:
            n_of_lines = n_of_lines +1
            number_of_b = 0
    return n_of_lines


def sum_matrix(image):
    """
    Calculates the sum of vlaues in the matrix

    Parameters
    ----------
    image : [8*8] array
        the image

    Returns
    -------
    double
        the sum of values in the whole matrix

    """
    return np.sum(image)       
                
            
    

"""
Dict of all feature functions
"""
feature_func_dict = {
        averageOfVerticalMiddleLine : "average value in vertical middle line"  ,
        longestVerticalLine: "longest vertical line",
        simetry : "simetry",
        sumOfValuesOnDiagonal : "sum of values on main diagonal",
        averageColorOfMiddle : "average color of middle pixels",
        averageRowIntersection : "average row intersection points",
        varMatrix : "variance of matrix",
        number_of_hori_lines: "number of horizontal lines",
        sum_matrix : "the sum of the matrix"
       
    }
       

#######################################################################################################################
#Display Functions:

def display1dPlot(x,y,num_1,num_2,name_of_feature,f_number):
    """
    Draws a plot of 1 feature

    Parameters
    ----------
    x : double array
        the values of the feauters on all images
    y : int array
        the classification
    num_1 : int
        the first number
    num_2 : int
        the second number
    name_of_feature : str
        the name of the feature we are using currently

    Returns
    -------
    None.

    """
    indecies_num_1 = np.where(y == num_1)
    indecies_num_2 = np.where(y == num_2)
    
    fig = plt.figure(figsize=(7,5))
    fig.suptitle(f"feature {f_number}: {name_of_feature},for {num_1} and {num_2} digits", fontsize=14)
    ax = plt.axes(yticklabels=[num_1,num_2],yticks= [num_1,num_2],ylabel="digits",xlabel=name_of_feature)

    plt.plot(x[indecies_num_1], y[indecies_num_1],'o',color = "purple")
    plt.plot(x[indecies_num_2], y[indecies_num_2],'o',color = "yellow")
    
    plt.show()
    
def display1Dclassifier(feature_function,data_image,data_target,num_1,num_2,f_number):
    """
    Display a plot for a single clissfier

    Parameters
    ----------
    feature_function : function
        the feature we are runing on
    data_image : array of [8*8] images
        the images themselfs.
    data_target : aray of ints
        array of the classification
    num_1 : int [0-9]
        the first number
    num_2 : int [0-9]
        the second number
    f_number: int
        the feature number

    Returns
    -------
    None.

    """
    feature_array = np.array(list(map(feature_function, data_image)))
    display1dPlot(feature_array,data_target,num_1,num_2,feature_func_dict[feature_function],f_number)
    

def display2dPlot(x_1,x_2,y,num_1,num_2,n_f_1,n_f_2):
    """
    draws a plot for 2 fetures

    Parameters
    ----------
    x_1 : array of doubles
        the feature score of the first feature
    x_2 : doubles
        the feature score of the second feature.
    y : array of ints [0-9]
        the classification
    num_1 : int
        first number
    num_2 : int
        second number
    n_f_1 : str
        name of first feature.
    n_f_2 : str
        name of second feature.

    Returns
    -------
    None.

    """
    indecies_num_1 = np.where(y == num_1)
    indecies_num_2 = np.where(y == num_2)
   
    fig = plt.figure(figsize=(7,5))
    fig.suptitle(f"{n_f_1} and {n_f_2} :for {num_1} and {num_2} digits", fontsize=12)
    ax = plt.axes(xlabel =n_f_1,ylabel = n_f_2 )

    plt.plot(x_1[indecies_num_1], x_2[indecies_num_1],'o',color = "purple")
    plt.plot(x_1[indecies_num_2], x_2[indecies_num_2],'o',color = "yellow")

    plt.show()
    
def display2Dclassifier(feature_function_1,feature_function_2, data_image,data_target,num_1,num_2):
    """
    Displays a 2d plot based on feature functions and the data

    Parameters
    ----------
    feature_function_1 : function
        function of the first feature.
    feature_function_2 : function
        function of the second feature.
    data_image : array of [8*8] images
        the number images themselfs.
    data_target : array of numbers [0-9]
        the classification of each image.
    num_1 : int
        first number we are intrested in
    num_2 : int
        second number we are intersted in

    Returns
    -------
    None.

    """
    feature_array_1 = np.array(list(map(feature_function_1, data_image)))
    feature_array_2 = np.array(list(map(feature_function_2, data_image)))
    display2dPlot(feature_array_1,feature_array_2,\
                  data_target,\
                  num_1,num_2,\
                  feature_func_dict[feature_function_1]\
                  ,feature_func_dict[feature_function_2])
    

def display3Dclassifier(feature_function_1, feature_function_2, feature_function_3, data_image, data_target,num_1,num_2):
    """
    Display a 3d feature plot based on feature functions and data
    Parameters
    ----------
    feature_function_1 : function
        function of the first feature.
    feature_function_2 : function
        function of the second feature.
    feature_function_3 : function
        function of the third feature.
    data_image : array of [8*8] images
        the number images themselfs.
    data_target : array of numbers [0-9]
        the classification of each image.
    num_1 : int
        first number we are intrested in
    num_2 : int
        second number we are intersted in

    Returns
    -------
    None.

    """
    feature_array_1 = np.array(list(map(feature_function_1, data_image)))
    feature_array_2 = np.array(list(map(feature_function_2, data_image)))
    feature_array_3 = np.array(list(map(feature_function_3, data_image)))
    display3dPlot(feature_array_1,feature_array_2,feature_array_3,\
                  data_target,\
                  feature_func_dict[feature_function_1],\
                  feature_func_dict[feature_function_2],\
                  feature_func_dict[feature_function_3],\
                  num_1,num_2)
    
def display3dPlot(x_1,x_2,x_3,y,n_f_1,n_f_2,n_f_3,num_1,num_2):
    """
    Draw a 3D plot based on 3 features and 2 numbers

    Parameters
    ----------
    x_1 : array of doubles
        the feature score of the first feature
    x_2 : array of doubles
        the feature score of the second feature
    x_3 : array of doubles
        the feature score of the third feature
    y : array of ints [0-9]
        the classification
    n_f_1 : str
        name of first feature.
    n_f_2 :  str
        name of second feature.
    n_f_3 : str
        name of third feature.
    num_1 : int
        first number
    num_2 : int
        second number

    Returns
    -------
    None.

    """
    indecies_num_1 = np.where(y == num_1)
    indecies_num_2 = np.where(y == num_2)
    fig = plt.figure()
    fig.suptitle(f"{n_f_1} ,{n_f_2},{n_f_3} :for {num_1} and {num_2} digits", fontsize=12)
    ax = fig.gca(projection='3d')
    ax.scatter(x_1[indecies_num_1], x_2[indecies_num_1], x_3[indecies_num_1],color="yellow")
    ax.scatter(x_1[indecies_num_2], x_2[indecies_num_2], x_3[indecies_num_2],color="purple")
    ax.set_xlabel(n_f_1)
    ax.set_ylabel(n_f_2)
    ax.set_zlabel(n_f_3)
    fig.show()


#####################################################################################
#Classifiers

def run_logistic_regression_0_1(digits,indecies_0_1):
    """
    Run logistic regresion on 0 and 1 images

    Parameters
    ----------
    digits : array of [8*8] images
            the number images themselfs.
    indecies_0_1 : arra of indecies
        the indecies of 0-1.

    Returns
    -------
    None.

    """
    feature_names = []
    feature_array_2 = np.array(list(map(sumOfValuesOnDiagonal, digits.images)))
    feature_array_3 = np.array(list(map(simetry, digits.images)))
    feature_array_5 = np.array(list(map(averageColorOfMiddle, digits.images)))
    feature_array_6 = np.array(list(map(averageRowIntersection, digits.images)))
    feature_names.append(feature_func_dict[averageOfVerticalMiddleLine])
    feature_names.append(feature_func_dict[sumOfValuesOnDiagonal])
    feature_names.append(feature_func_dict[simetry])
    feature_names.append(feature_func_dict[longestVerticalLine])
    feature_names.append(feature_func_dict[averageColorOfMiddle])
    feature_names.append(feature_func_dict[averageRowIntersection])

    LR(digits.target[indecies_0_1],feature_names,feature_array_2[indecies_0_1],\
                                   feature_array_6[indecies_0_1],\
                                   feature_array_3[indecies_0_1],\
                                   feature_array_5[indecies_0_1])
        
def run_logistic_regression_all(digits):
    """
    Run logistic regression on all numbers

    Parameters
    ----------
    digits : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    feature_array_1 = np.array(list(map(averageOfVerticalMiddleLine,digits.images)))
    feature_array_2 = np.array(list(map(sumOfValuesOnDiagonal, digits.images)))
    feature_array_3 = np.array(list(map(simetry, digits.images)))
    feature_array_4 = np.array(list(map(longestVerticalLine, digits.images)))
    feature_array_5 = np.array(list(map(averageColorOfMiddle, digits.images)))
    feature_array_6 = np.array(list(map(averageRowIntersection, digits.images)))
    feature_array_7 = np.array(list(map(sum_matrix, digits.images)))
    feature_array_8 = np.array(list(map(varMatrix, digits.images)))
    feature_array_9 = np.array(list(map(number_of_hori_lines, digits.images)))
    
    feature_names = []
    feature_names.append(feature_func_dict[averageOfVerticalMiddleLine])
    feature_names.append(feature_func_dict[sumOfValuesOnDiagonal])
    feature_names.append(feature_func_dict[simetry])
    feature_names.append(feature_func_dict[longestVerticalLine])
    feature_names.append(feature_func_dict[averageColorOfMiddle])
    feature_names.append(feature_func_dict[averageRowIntersection])
    feature_names.append(feature_func_dict[sum_matrix])
    feature_names.append(feature_func_dict[varMatrix])
    feature_names.append(feature_func_dict[number_of_hori_lines])
    
    LR(digits.target,feature_names,feature_array_1,
                          feature_array_2,
                          feature_array_3,\
                          feature_array_4,\
                          feature_array_5,\
                          feature_array_6,\
                          feature_array_7,\
                          feature_array_8,\
                          feature_array_9)
                                
                                   
                                 
    
def LR(y,feature_names,*args):
    """
    Run logistic regression based on y and feature argumets

    Parameters
    ----------
    y : array of ints [0-9]
        the true classifcation.
    *args : arrays of feauters
        all the feature arrays

    Returns
    -------
    None.

    """
    feature_str = ""
    for name in feature_names:
        feature_str = feature_str + name
        feature_str = feature_str + ", "
        
    X = np.column_stack(args)
    
    # scaling the values for better classification performance
    X_scaled = preprocessing.scale(X)
    # the predicted outputs
    Y = y
    # Training Logistic regression
    logistic_classifier = linear_model.LogisticRegression(solver='lbfgs')
    logistic_classifier.fit(X_scaled, Y)
    # show how good is the classifier on the training data
    expected = Y
    predicted = logistic_classifier.predict(X_scaled)
    
    print("Logistic regression using %s features:\n%s\n" % (
    feature_str,
    metrics.classification_report(
    expected,
    predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    # estimate the generalization performance using cross validation
    predicted2 = cross_val_predict(logistic_classifier, X_scaled, Y, cv=10)
        
    print("Logistic regression using %s features cross \
    validation:\n%s\n" % ( \
    feature_str,  
    metrics.classification_report( \
    expected, \
    predicted2)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted2))
   
    
      

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()



###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)



#Main
###############################################################################
#Question number 19:
getWrongClassificationsFromDS(X_test,predicted,y_test)


#1 feature classifers:
indecies_0_1 = extractSpecificIndexDigitsFromDS(0,1,digits)

for i,feature_func in enumerate(feature_func_dict):
    display1Dclassifier(feature_func,digits.images[indecies_0_1],digits.target[indecies_0_1],0,1,i+1)
    
#2 feature classifiers
display2Dclassifier(averageRowIntersection,simetry, digits.images[indecies_0_1],digits.target[indecies_0_1],0,1)
display2Dclassifier(averageColorOfMiddle,simetry, digits.images[indecies_0_1],digits.target[indecies_0_1],0,1)
display2Dclassifier(longestVerticalLine,averageColorOfMiddle, digits.images[indecies_0_1],digits.target[indecies_0_1],0,1)

#3 feature classifiers
display3Dclassifier(simetry, averageRowIntersection, averageColorOfMiddle,\
                    digits.images[indecies_0_1], digits.target[indecies_0_1],0,1)
display3Dclassifier(simetry, averageRowIntersection, sumOfValuesOnDiagonal,\
                    digits.images[indecies_0_1], digits.target[indecies_0_1],0,1)
display3Dclassifier(simetry, averageRowIntersection, longestVerticalLine,\
                   digits.images[indecies_0_1], digits.target[indecies_0_1],0,1)

#Logistic regression on 0 and 1
run_logistic_regression_0_1(digits,indecies_0_1)

run_logistic_regression_all(digits)


 


 
                


