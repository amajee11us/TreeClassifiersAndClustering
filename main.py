from dataloader import LoadQ1Data, MNISTloader
from classifier import TreeClassifier
from KMeans import KMeans
from utils import *
import argparse
import math
from parameterFactory import pickBestParams
from itertools import product

def parse_args():
    parser = argparse.ArgumentParser(
        description='Assignment 3 : KMeans and Tree Classifiers.')

    # General parser
    parser.add_argument('-q',
                        '--question',
                        dest='question',
                        type=int,
                        default=1,
                        help='The question number of the problem.')
    parser.add_argument('--k',
                        default=2,
                        type=int,
                        help='k value for clustering. Example : 2, 5, 10, 15, 20 etc.')
    # parser.add_argument('--datapoints',
    #                     default=500,
    #                     type=int,
    #                     help='Number of datapoints in the dataset. Example : 1000, 1800 etc.')
    parser.add_argument('--model_name',
                        default='dtree',
                        type=str,
                        help='Name of the model for which experiment is triggered. Example : dtree, bagging etc.')
    parser.add_argument('--no_tuning',
                        default=False,
                        type=bool,
                        help='Choose to perform or ignore tuning the model. Only applicable for question 2')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if args.question == 1:
        '''
        Tree Classifier Problem
        '''
        base_data_path = "data/all_data_q1"
        clauses = [500, 1000, 1500, 1800] 
        datapoints = [100, 1000, 5000] 

        all_comb_c_d = list(product(clauses, datapoints))

        # Store in results file
        result_file = open("results_{}.txt".format(args.model_name), "w")

        # Run for all possible combinations of clauses and datapoints        
        for clause, data_point in all_comb_c_d:
            print("### Running the Model for Model : {}, Clauses : {}, Datapoints : {} ###".format(args.model_name, clause, data_point))

            print("Loading datasets ...")
            # load training data
            trainDataset = LoadQ1Data(base_path=base_data_path, clause=clause, datapoints=data_point, split='train')

            # load test data
            valDataset = LoadQ1Data(base_path=base_data_path, clause=clause, datapoints=data_point, split='valid')

            # load test data
            testDataset = LoadQ1Data(base_path=base_data_path, clause=clause, datapoints=data_point, split='test')
            print("Done.")

            # Data loader class for MNIST dataset 
            # print("Loading MNIST dataset ...")
            # data_loader = MNISTloader(dataset_name = args.dataset_name, num_classes=10)
            # print("Done.")
            
            print("Loading Model ...")
            model = TreeClassifier(model_name=args.model_name)
            print("Done.")

            X_train, Y_train = trainDataset.get_data()
            X_test, Y_test = testDataset.get_data()
            X_val, Y_val = valDataset.get_data()

            if args.no_tuning:
                print("Starting Model Trainval ...")
                final_params = pickBestParams(args.model_name)            
            else:
                print("Starting Model Tuning ...")
                final_params = model.tune(X_train, Y_train, 
                                        X_val, Y_val)
                print("Done.")
            
            X_train, Y_train = trainDataset.combine_rawData(X_train, Y_train, 
                                                            X_val, Y_val)
            predictions = model.trainval(X_train, Y_train, 
                                        X_test, Y_test,
                                        final_params)
            print("Done.")

            metrics = getPerformanceScores(predictions, Y_test, "classifier")
            printResults(result_row=[x for x in metrics], table_columns=model.pred_metrics)

            # Store all the variables in results file
            output_string = "{},{}".format(clause, data_point)
            for item in final_params:
                output_string += ",{}".format(str(item))
            output_string += ",{}\n".format(str(round(metrics[-1],3)))
            print(output_string)
            result_file.write(output_string)

        result_file.close()
    else:
        '''
        K-Means Clustering - Image Compression
        '''
        image_path = ["data/Koala.jpg", "data/Penguins.jpg"]
        for image in image_path:
            image_name = image.split('.')[0].split('/')[-1]
            kmeansClf = KMeans(image_path= image,k = args.k)
            output_c_ratios = kmeansClf.kmeans(iterations=10, image_name=image_name) # performs kMeans
            print("Results for K = {}".format(args.k))
            
            printResults(result_row=output_c_ratios, 
                         table_columns=[str(x + 1) for x in range(10)])

        
