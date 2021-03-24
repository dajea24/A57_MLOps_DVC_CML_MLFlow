# ProjectA57
Machine Learning Operations: CML-MLFlow-DVC integration for Linear Regression


### Modifications apportées au fichier train.py

     
    *** # Write scores to a file
        with open("metrics.txt", 'w') as outfile:
            outfile.write("MSE:  {0:2.1f} \n".format(rmse))
            outfile.write("MAE:  {0:2.1f} \n".format(mae))
            outfile.write("R2: {0:2.1f}\n".format(r2))
            
        # save the model to disk
    *** filename = 'linear_model.pkl'
    *** pickle.dump(lr, open(filename, 'wb'))
