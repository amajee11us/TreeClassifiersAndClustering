# Tree Classifiers and K-Means Clustering
A simple codebase to implement Tree Classifiers and K-Means clustering.

# Setup Instructions
1. Data Preparation 
Pertains only to question 1. Place the .zip files downloaded from eLearning into the data directory. Extract the contents using the command below -

```
cd data
unzip -v hw3_part1_data.zip # You can ommit the -v if you do not want verbose printed
unzip -v hw3_part2_data.zip
```

2. Environment setup
Common packages like numpy, pandas and scikit-learn are required for this codebase to run. If you are using anaconda please use the following command to create the environment.

```
conda env create -f environment.yml
```

3. Question 1 - Tree Classifiers
    a. To execute model tuning on Decision Tree Classifier run the below command

        ```python main.py -q 1 --model_name dtree```

    b. To execute model tuning on Bagging Ensemble Classifier run the below command

        ```python main.py -q 1 --model_name bagging```

    c. To execute model tuning on Gradient Boosting Classifier run the below command

        ```python main.py -q 1 --model_name gradBoost```
    
    d. To execute model tuning on Random Forest Classifier run the below command

        ```python main.py -q 1 --model_name randomForest```
    
    e. Model tuning will be automatically followed by execution of the model on the best performing model. If you want to explicitly run a model with the best set of parameters execute the below command

        ```python main.py --question 1 --model_name <name_of_model> --no_tuning```

4. Question 1 - K-Means 
To execute the tree classifier algorithms run the below commands

python main.py --question 1 --k <2/5/10/15/20>

# References 
After the assignment is graded I plan on releasing the codebase to public git under this repository - https://github.com/amajee11us/TreeClassifiersAndClustering

### Author - Anay Majee (anay.majee@utdallas.edu)