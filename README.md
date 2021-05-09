# BIOF509-Final-Project
This is the repository for my BIOF509 Final Project  
Any Questions, please contact me: joshua.breidenbach@rockets.utoledo.edu

File Name: BIOF509 - Applied Machine Learning - Final Project - Joshua Breidenbach.ipynb  
This is a python notebook file that can be used in Jupyter Notebook  
Data: The instructor James said that not having access to this dataset was okay.  
  
The 'Initial Imports' are important to run first.  
Steps taken in the 'Preprocessing' sections: (Import, removing unwanted features, dealing with missing data by removing some more features and then patients, balancing the data, and finally setting the target and label encoding it) are very specific to my dataset which is a clinical trial dataset which you need special access to.  
  
<ins>Scaling:</ins> 
However, 'Scaling' is a generalized class object that should work with any clean data not missing any values.  
**To scale:**  
name_of_scaled_data = Preprocessing(df_to_be_scaled)  
name_of_scaled_data.scale()  
This will returen a scaled array of arrays of your data.  
  
#Supervised Machine Learning  
There are multiple machine learning class objects that can easily be used with any scaled data.  
  
<ins>Support Vector Machine:</ins>  
In the below example my scaled data is "name_of_scaled_data.scale()" because it's coming from the scaling class object above.  
**To fit and predict:**  
name = SVM(name_of_scaled_data.scale(), target, name_of_unscaled_df)  
name.fit_and_predict()  
**To build a confusion matrix from this model:**  
name.cm()  
**To build a plot of the feature importances:**  
name.feature_importance()  
  
<ins>Random Forest Classifier:</ins>   
In the below example my scaled data is "name_of_scaled_data.scale()" because it's coming from the scaling class object above.  
You can adjust the number of trees used in the forest model by changing the parameter "#ofTrees_in_forest" below.  
**To fit and predict:**  
name = RFC(name_of_scaled_data.scale(), target, name_of_unscaled_df, #ofTrees_in_forest)  
name.fit_and_predict()  
**To build a confusion matrix from this model:**  
name.cm()  
**To build a plot of the feature importances:**  
name.feature_importance()  
  
<ins>Random Forest Regressor:</ins>   
In the below example my scaled data is "name_of_scaled_data.scale()" because it's coming from the scaling class object above.  
You can adjust the number of trees used in the forest model by changing the parameter "#ofTrees_in_forest" below.  
REMEMBER: for a regression approach, the target will not be a class, but will be actual numerical float values.  
**To fit and predict:**  
name = RFR(name_of_scaled_data.scale(), target, #ofTrees_in_forest)  
name.fit_and_predict()  
  
<ins>Artificial Neural Network::</ins>     
In the below example my scaled data is "name_of_scaled_data.scale()" because it's coming from the scaling class object above.  
This network is built using TensorFlow through Keras, so these packages will need to be installed and imported.  
This class function builds 3 artificial neural networks which run 20 epochs.  
Model 1 has 1 hidden layer of 1000 neurons.  
Model 2 has 3 hidden layers of 1000, 10000, and 1000 neurons.  
Model 3 has 1 hidden layer of 10000 neurons.  
**To fit:**  
name = NN(name_of_scaled_data.scale(), target, #of_Input_Features)  
name.fit()  
**To predict:**  
First, you can predict using the training data and get a summary.  
print("Best model for train data:")  
print(NN_df.pred_train().summary())  
Then, you can predict using the testing data and get a summary.  
print("Best model for test data:")  
print(NN_df.pred_test().summary())  
  
Please see the code for many more notes and comments.
