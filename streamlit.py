import joblib
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score
import imblearn
from imblearn.over_sampling import SMOTE

# Load the data
dados = pd.read_csv('./indian_liver_patient.csv')

# Sample data
dados.sample(10)


# 
# 
# 
# EXPLORATORY DATA ANALYSIS

# In[2]:


# Data types
print(dados.dtypes)

# Categorical variables
print(dados.dtypes[dados.dtypes == 'object'])

# Non-categorical variables
print(dados.dtypes[dados.dtypes != 'object'])

# Exploration of Numerical Variables
dados.describe()

# Plot 
dados.hist(figsize = (15, 15), bins = 10)
plt.show()

# Apparently there is an outlier in the variables "Alamine_Aminotransferase" and "Aspartate_Aminotransferase"
# Due to the fact that the maximum value is much higher than the average value.
# The dataset column (target variable) has '1' for liver disease and '2' for no liver disease.
# Let's adjust the variable by putting values that are easier to interpret. The negative class (does not have the disease) will be zero.

# Function to adjust target variable
def ajusta_var(x):
    if x == 2:
        return 0
    return 1

# Apply the function
dados['Dataset'] = dados['Dataset'].map(ajusta_var)

# Let's adjust the target variable name
dados.rename({'Dataset':'Target'}, axis = 'columns', inplace = True)


# ATTRIBUTE ENGINEERING

# In[3]:


# Select only numeric columns
dados_numeric = dados.select_dtypes(include=['float64', 'int64'])

# Print the columns
print(dados_numeric.columns)

# Correlation between variables
correlation_matrix = dados_numeric.corr()

# Print the correlation matrix
print(correlation_matrix)

# Exploration of the Categorical Variable
dados.describe(include = ['object'])

# Plot
sns.countplot(data = dados, x = 'Gender', label = 'Count')

# Value counts
M, F = dados['Gender'].value_counts()

# Print
print('Number of male patients: ', M)
print('Number of female patients: ', F)



# In[4]:


# Let's take advantage of this and transform the categorical variable into its numeric representation using label encoding.
# In addition to reducing work later, it will make it easier to create charts to follow.
# Function for label encoding
def encoding_func(x):
    if x == 'Male':
        return 0
    return 1

# Apply the function
dados['Gender'] = dados['Gender'].map(encoding_func)
dados.sample(5)

# Checking the Relationship Between Attributes
dados.corr()

# Set the background style
sns.set_style('darkgrid')  

# Facetgrid
sns.FacetGrid(dados, hue = 'Target').map(plt.scatter, 'Total_Bilirubin', 'Direct_Bilirubin').add_legend()

# Set the background style
sns.set_style('darkgrid')  

# Facetgrid
sns.FacetGrid(dados, hue = 'Gender').map(plt.scatter, 'Total_Bilirubin', 'Direct_Bilirubin').add_legend()

# Set the background style
sns.set_style('whitegrid') 

# Facetgrid
sns.FacetGrid(dados, hue = 'Target').map(plt.scatter, 'Total_Bilirubin', 'Albumin').add_legend()

# Set the background style
sns.set_style('whitegrid') 

# Facetgrid
sns.FacetGrid(dados, hue = 'Gender').map(plt.scatter, 'Total_Bilirubin', 'Albumin').add_legend()

# Checking for Missing Values and Duplicate Records
# Checking for missing values
print(dados[dados.isnull().values])

# Checking for duplicate records (complete cases)
# Complete cases also refer to lines where there are no missing values
print(dados[dados.duplicated(keep = False)])


# In[5]:


# Handling Duplicate Records
print(dados.shape)

# Remove duplicate records (remove one of the duplicates)
dados = dados.drop_duplicates()
print(dados.shape)

# Handling Outliers
dados.describe()

# Boxplot
sns.boxplot(dados.Alamine_Aminotransferase)

# Are the extreme values really outliers? Frequency count by value
dados.Alamine_Aminotransferase.sort_values(ascending = False).head()

# Boxplot
sns.boxplot(dados.Aspartate_Aminotransferase)

# Frequency count by value
dados.Aspartate_Aminotransferase.sort_values(ascending = False).head()

# Keep only records where the value is less than or equal to 3000
dados = dados[dados.Aspartate_Aminotransferase <= 3000]

# Boxplot
sns.boxplot(dados.Aspartate_Aminotransferase)

# Frequency count by value
dados.Aspartate_Aminotransferase.sort_values(ascending = False).head()

# Keep only records where the value is less than or equal to 2500
dados = dados[dados.Aspartate_Aminotransferase <= 2500]
dados.describe()

# Handling Missing Values. Check for missing value
dados.isnull().values.any()

# Check how many columns have missing value
dados.isnull().values.any().sum()

# List missing values
print(dados[dados.isnull().values])

# Drop records with missing values in any column (any)
dados = dados.dropna(how = 'any')  

# List missing values
print(dados[dados.isnull().values])


# PRE-PROCESSING DATA

# Split into Training and Test
dados.head()

# Create a separate object for the target variable
y = dados.Target

# Create a separate object for input variables
X = dados.drop('Target', axis = 1)

# Split into training and test data with stratified sampling
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y,
test_size = 0.25, random_state = 1234, stratify = dados.Target)
len(X_treino)
len(X_teste)

# Print do shape
print(X_treino.shape, X_teste.shape, y_treino.shape, y_teste.shape)
X_treino.head(2)

# Class Balancing
# As it stands, we have a lot more information about the variable target(1) than the variable(0)
# With this, we will be giving the model many more examples of the first class than the second class
# Making it learn much more about a class than about another, generating a biased model.
y_treino.value_counts()

# A first strategy would be to reduce the majority class records by removing some records from Class 1
# This strategy can greatly reduce the size of the dataframe, thus having fewer examples to train the model
# Another strategy would be the technique of oversampling and increasing the number of examples of the minority class
# In order to detect the pattern of the records of the class (0), and create synthetic data with the same pattern.
# Increasing with this the amount of lines of the minority class.
over_sampler = SMOTE(k_neighbors = 2)

# Explain why class balancing is done with training data only.
# Apply oversampling (should only be done with training data)
X_res, y_res = over_sampler.fit_resample(X_treino, y_treino)
len(X_res)
len(y_res)
y_res.value_counts()

# Set the training dataset name to X
X_treino = X_res

# Set the training dataset name to y
y_treino = y_res

# Data standardization
# The goal is to resize the variables so that they have properties of
# A normal distribution with mean equal to zero and standard deviation equal to one.
X_treino.head()

# Calculate mean and standard deviation of training data
treino_mean = X_treino.mean()
treino_std = X_treino.std()
print(treino_mean)
print(treino_std)

# Standardization
X_treino = (X_treino - treino_mean) / treino_std
X_treino.head()
X_treino.describe()

# We use training mean and deviation to standardize the test dataset
X_teste = (X_teste - treino_mean) / treino_std
X_teste.head()



import streamlit as st
import joblib
import numpy as np

# Load the best model from disk
melhor_modelo = joblib.load('neural_network_model.pkl')

def main():
    st.title("Liver Disease Prediction")

    st.write("Enter patient details:")
    age = st.number_input("Age")
    gender = st.radio("Gender", ["Female", "Male"])
    total_bilirubin = st.number_input("Total Bilirubin")
    direct_bilirubin = st.number_input("Direct Bilirubin")
    alkaline_phosphotase = st.number_input("Alkaline Phosphotase")
    alamine_aminotransferase = st.number_input("Alamine Aminotransferase")
    aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase")
    total_proteins = st.number_input("Total Proteins")
    albumin = st.number_input("Albumin")
    albumin_globulin_ratio = st.number_input("Albumin and Globulin Ratio")

    gender_code = 0 if gender == "Female" else 1

    novo_paciente = [age, gender_code, total_bilirubin, direct_bilirubin, alkaline_phosphotase, alamine_aminotransferase, 
                     aspartate_aminotransferase, total_proteins, albumin, albumin_globulin_ratio]

    # Convert object to array
    arr_paciente = np.array(novo_paciente)

    # Assuming 'treino_mean' and 'treino_std' are the mean and standard deviation used during training
    # We use training mean and deviation to standardize new data
    arr_paciente = (arr_paciente - treino_mean) / treino_std

    # Convert object to array
    arr_paciente = np.array(arr_paciente)

    # Standardized patient data (exactly how the model expects to receive the data)
    st.write("Standardized Patient Data:", arr_paciente)

    # Class predictions
    pred_novo_paciente = melhor_modelo.predict(arr_paciente.reshape(1, -1))

    # Check the value and print the final result
    if pred_novo_paciente == 1:
        # Run the model three more times
        for _ in range(3):
            # Assuming 'arr_paciente' is already standardized
            pred_novo_paciente = melhor_modelo.predict(arr_paciente.reshape(1, -1))
            
            if pred_novo_paciente == 0:
                # If any subsequent run predicts 'no', print 'Treatable' and break the loop
                st.write('Treatable')
                break
        st.write('This patient must have liver disease!')
    else:
        st.write('This patient must not have liver disease!')

if __name__ == "__main__":
    main()