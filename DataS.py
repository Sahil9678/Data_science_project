# Miceforest is a Python tool that fills in missing values (like blanks or NaNs in your data) using a smart method called Random Forest.
# When your dataset has some missing values, you can't train a model properly. 
# Miceforest looks at the patterns in the data and guesses the missing values in a smart way.

# 🔧 How to use it?
# 1.Import the library.
# 2.Create a kernel with your data.
# 3.Run the imputation.
# 4.Get the completed data.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from miceforest import ImputationKernel
import missingno as msno
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv('METABRIC_RNA_Mutation.csv')
data.drop(["patient_id"],axis=1,inplace = True)
data.head()
data.describe(include='all')

print(data.isnull().sum())

#  ---------------------------------------------------------------------------------------------------------------------------------------------- 
# we are Looking at the data in charts and graphs to find patterns in medical and genetic information that 
# might affect how long cancer patients survive.

# list of clinical features (medical details) from the dataset
columns = ['age_at_diagnosis', 'lymph_nodes_examined_positive','mutation_count','nottingham_prognostic_index', 'overall_survival_months', 'tumor_size' ]   

fig, axs = plt.subplots(3, 2, figsize=(20, 7))
fig.suptitle('Clinical Data Analysis')
    
# adding light background with grid lines
sns.set(style="whitegrid")
    
for i,ax in zip(data[columns].columns,axs.flatten()):
    sns.histplot(data[i][data['overall_survival']==1], color='g', label = 'Survived',ax=ax)
    sns.histplot(data[i][data['overall_survival']==0], color='r', label = 'Died',ax=ax)
    ax.legend(loc='best')
plt.tight_layout()
# plt.show()

# list of gene names from the dataset
columns = ['pik3ca','tp53','muc16','ahnak2','kmt2c','syne1','gata3','map3k1','ahnak','dnah11','cdh1','dnah2','kmt2d','ush2a','ryr2']   

fig, axs = plt.subplots(3, 5, figsize=(15, 10))
fig.suptitle('Survival of patients with some of gene mutations.')

for i,ax in zip(data.loc[:,columns].columns,axs.flatten()):
    sns.histplot(data[i][data['overall_survival']==0], color='g', label = 'Survived',ax=ax)
    sns.histplot(data[i][data['overall_survival']==1], color='r', label = 'Died',ax=ax)
    ax.legend(loc='best')
plt.tight_layout()
# plt.show()

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 7))
sns.histplot(data['tumor_size'], bins=20, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Tumor Size')
axes[0].set_xlabel('Tumor Size')
axes[ 0].set_ylabel('Frequency')

# create a boxplot to quickly spot unusual values
# here is the diagram:- https://support.minitab.com/en-us/minitab/media/generated-content/images/boxplot_pulse.png
sns.boxplot(data['mutation_count'], ax=axes[1])
axes[1].set_title('Box Plot of Mutation Count')
axes[1].set_xlabel('Mutation Count')

# creates a scatter plot to visually explore patterns in the data.
# for example, if survival time goes down as tumor size increases.

# here is the diagram:- https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Scatter_diagram_for_quality_characteristic_XXX.svg/640px-Scatter_diagram_for_quality_characteristic_XXX.svg.png
sns.scatterplot(data=data, x='tumor_size', y='overall_survival_months', ax=axes[2])
axes[2].set_title('Tumor Size vs. Overall Survival')
axes[2].set_xlabel('Tumor Size')
axes[2].set_ylabel('Overall Survival (Months)')

plt.tight_layout()
# plt.show()

data_survival = data[data['death_from_cancer'].notnull()].copy()
clinical_data = data_survival[data_survival.columns[:30]].copy()

# Lets start handling missing values. There are few types of missing data:
# Missing Completely at Random (MCAR) - In MCAR, the probability of a value being missing is unrelated to any observed or unobserved data.
# Missing at Random (MAR) - In MAR, the probability of a value being missing depends only on observed data and not on unobserved data.
# Missing Not at Random (MNAR) - In MNAR, the probability of a value being missing depends on unobserved data.
fig, axes = plt.subplots(2, 1, figsize=(10, 10))
ax1 = msno.matrix(clinical_data, ax=axes[0],sparkline=False)
ax2 = msno.heatmap(clinical_data, ax=axes[1])
# plt.show()

# data is definitely not missing completely at random. In that case we can use Multiple imputation to deal with missing values.

# Converts the death_from_cancer column (which has yes/no or true/false values) into 0s and 1s.
# This is needed because machine learning models only work with numbers, not text.
encoder = LabelEncoder()
clinical_data['death_from_cancer']=encoder.fit_transform(clinical_data['death_from_cancer'])
# read readme.md
clinical_data_encoded = pd.get_dummies(clinical_data)
# print("clinical_data_encoded ",clinical_data_encoded)

# replce all these value "!@#$%^&*(){}\[\];:,./<>?\\|`~\=_\'" to "_" (underscore)
clinical_data_encoded.columns = [re.sub(r'[!@#$%^&*(){}\[\];:,./<>?\\|`~\=_\']', '_', col) for col in clinical_data_encoded.columns]
# read readme.md
clinical_data_encoded.describe()

clinical_data_encoded = clinical_data_encoded.reset_index(drop=True)

# we are handling missing data in a smart way, so machine learning models can train on a complete dataset without errors or bias.
mice_kernel = ImputationKernel(data = clinical_data_encoded,random_state = 42)
mice_kernel.mice(2)
data_full = mice_kernel.complete_data()
data_full.describe()

#  ---------------------------------------------------------------------------------------------------------------------------------------------- 
# We are Looking at the data in charts and graphs to find patterns in medical and genetic information that 
# might affect how long cancer patients survive and also fixing the missing data values in the dataset.
#  ---------------------------------------------------------------------------------------------------------------------------------------------- 


columns_to_keep = data_full.columns != 'death_from_cancer'
X = data_full.loc[:,columns_to_keep]
y = data_full['death_from_cancer']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# ---------------------------------Function to train and evaluate multiple models using cross-validation---------------------------------
def train_models_simple(models, X, y, cv):
    
    # To store model names
    model_names = []

    # To store average accuracy of each model
    accuracy_scores = []    

    # ---------------------------------Loop through each model in the list---------------------------------
    for name, model in models:
        print(name)

        # here we are going to perform cross-validation with accuracy as the scoring metric
        result = cross_validate(model, X, y, cv=cv, scoring='accuracy')

        # we also need to calculate mean accuracy score
        mean_accuracy = result['test_score'].mean()

        # then we will store the name and accuracy
        model_names.append(name)
        accuracy_scores.append(mean_accuracy)

        # Print accuracy of the current model
        print("name -",name)
        print("mean_accuracy-", mean_accuracy)

    # === PLOTTING ACCURACY SCORES ===

    # Create a bar chart to compare model accuracies
    sns.barplot(x=model_names, y=accuracy_scores)
    plt.title('Model Accuracy Comparison')   # Chart title
    plt.ylabel('Accuracy')                   # Y-axis label
    plt.xticks(rotation=45)                  # Rotate model names for readability
    plt.ylim(0, 1)                           # Accuracy range (0 to 1)
    plt.tight_layout()                       # Adjust layout
    plt.show()                               # Show the chart

    # Return model names and their accuracies
    return model_names, accuracy_scores

# We also gave a defined list of models with names
models = [
        ("LogisticRegression", LogisticRegression()),
        ("RandomForest", RandomForestClassifier(max_depth=10, random_state=42)),
        ("KNeighbors", KNeighborsClassifier()),
        ("DecisionTree", DecisionTreeClassifier(max_depth=10, random_state=42)),
        ("NaiveBayes", GaussianNB())
]

# Now just train and evaluate models using 5-fold cross-validation
train_models_simple(models, X, y, cv=5)