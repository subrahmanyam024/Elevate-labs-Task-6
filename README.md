# Elevate-labs-Task-6

KNN Classification for Iris Dataset
Overview
This Python script implements a K-Nearest Neighbors (KNN) classification model for the Iris dataset. It trains a KNN classifier, evaluates its performance for different K values, and visualizes the results, including decision boundaries. The code is designed to demonstrate instance-based learning, feature normalization, model evaluation, and visualization of classification boundaries.
Features

Data Preprocessing: Loads the Iris dataset, encodes class labels, and normalizes features using StandardScaler.
KNN Classification: Trains KNN models with K values from 1 to 20, selects the best K based on test accuracy.
Evaluation: Computes accuracy and generates a confusion matrix for the best model.
Visualizations:
Plot of accuracy vs. K values.
Confusion matrix heatmap.
2D decision boundary plot using Sepal Length and Sepal Width, with training points and a legend.



Dependencies

Python 3.x
Required libraries:
pandas: For data loading and manipulation.
numpy: For numerical operations.
scikit-learn: For KNN classification, data splitting, scaling, and metrics.
matplotlib: For plotting visualizations.
seaborn: For enhanced confusion matrix visualization.



Install dependencies using:
pip install pandas numpy scikit-learn matplotlib seaborn

Dataset

File: Iris.csv (must be in the working directory).
Content: Contains 150 samples of Iris flowers with four features (Sepal Length, Sepal Width, Petal Length, Petal Width) and three classes (Iris-setosa, Iris-versicolor, Iris-virginica).
Format: CSV with columns Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species.

Usage

Prepare the Environment:

Ensure Iris.csv is in the same directory as the script.
Install required libraries (see Dependencies).


Run the Script:

Save the code as knn_iris_classification_inline.py.
Execute in a Python environment (e.g., terminal, Jupyter notebook):python knn_iris_classification_inline.py


For Jupyter notebooks, include %matplotlib inline at the start to display plots.


Expected Outputs:

Console Output:
Best K value (e.g., Best K value: 1).
Accuracy of the best model (e.g., Accuracy with best K: 1.000).


Plots:
Accuracy vs. K Plot: Shows test accuracy for K values from 1 to 20.
Confusion Matrix: Heatmap showing true vs. predicted classes.
Decision Boundary Plot: 2D visualization of KNN decision boundaries using Sepal Length and Sepal Width, with training points and a legend for the three Iris classes.





Code Structure

Data Loading and Preprocessing:

Loads Iris.csv using pandas.
Encodes species labels (0: Setosa, 1: Versicolor, 2: Virginica).
Splits data into 80% training and 20% testing sets.
Normalizes features using StandardScaler.


KNN Training and Evaluation:

Trains KNN models for K=1 to 20 using KNeighborsClassifier.
Evaluates accuracy on the test set and selects the best K.
Computes a confusion matrix for the best model.


Decision Boundary Visualization:

Creates a 2D mesh grid over scaled Sepal Length and Sepal Width.
Uses mean values for Petal Length and Petal Width to match the four-feature input required by the KNN model.
Plots decision boundaries using plt.contourf with light colors (red, green, blue for Setosa, Versicolor, Virginica).
Overlays training points in their original scale with bold colors.
Adds a legend using empty scatter plots to represent each class.



Notes on Decision Boundary Visualization

The KNN model is trained on four features, but the decision boundary is visualized in 2D using Sepal Length and Sepal Width.
To handle the four-feature requirement, mean values of scaled Petal Length and Petal Width are used for grid point predictions.
The decision boundary is non-linear, reflecting KNN’s instance-based learning, and shows regions where the model predicts each class.
The plot includes:
Colored regions for decision boundaries (light red, green, blue).
Training points in bold colors (red, green, blue) with black edges.
X-axis: Sepal Length (cm), Y-axis: Sepal Width (cm).
Legend: Shows class names (Iris-setosa, Iris-versicolor, Iris-virginica) with corresponding colors.



Troubleshooting

No Visualization:
Ensure %matplotlib inline is included in Jupyter notebooks.
Verify all dependencies are installed.
Check that Iris.csv is in the working directory.
Save the plot to a file for debugging:plt.savefig('decision_boundaries.png')




Errors:
FileNotFoundError: Ensure Iris.csv exists.
ValueError (feature mismatch): The code uses four features for predictions to match the model’s training.


Blank Plot: Ensure the plotting backend is configured (e.g., matplotlib.use('TkAgg') for non-Jupyter environments).

Example Output

Console:Best K value: 1
Accuracy with best K: 1.000


Plots:
Accuracy vs. K: Line plot with markers showing accuracy for each K.
Confusion Matrix: Heatmap with counts of true vs. predicted classes.
Decision Boundary: 2D plot with colored regions, scattered training points, and a legend.

