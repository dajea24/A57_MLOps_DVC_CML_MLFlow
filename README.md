# ProjectA57
Machine Learning Operations: CML-MLFlow-DVC integration for Linear Regression


### Modifications apportées au fichier train.py

with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
 
        predicted_qualities = lr.predict(test_x)
 
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
 
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
     
    *** # Write scores to a file
        with open("metrics.txt", 'w') as outfile:
            outfile.write("MSE:  {0:2.1f} \n".format(rmse))
            outfile.write("MAE:  {0:2.1f} \n".format(mae))
            outfile.write("R2: {0:2.1f}\n".format(r2))
            
        # save the model to disk
    *** filename = 'linear_model.pkl'
    *** pickle.dump(lr, open(filename, 'wb'))

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
 
        mlflow.sklearn.log_model(lr, "model")
