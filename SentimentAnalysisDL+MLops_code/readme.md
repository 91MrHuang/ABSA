# POC for Aspect Level Sentiment Analysis

## Spin up local MLflow Server

* Install MLflow by 
  ```
  pip install mlflow
  ``` 

* Install and start mysql service locally https://dev.mysql.com/downloads/

* create a schema named sth like "mlflow" in mysql

* run
  ```
  mlflow server --backend-store-uri mysql+pymysql://root:19830728@localhost/mlflow_1.30.0 --default-artifact-root /opt/jagundi/mlruns
  ```
  
Then you can access the UI at http://localhost:5000/

More details here https://www.mlflow.org/docs/latest/quickstart.html




* Spin up service locally
  * run the following cmd to spin up aspect extraction service
    ```
    # bert
    mlflow models serve -m /opt/jagundi/mlruns/3/5dced4581b7a4cf590f371391d2ed571/artifacts/aspect-extraction -p 1188 -t 180 --env-manager=local
    
    # distilbert
    mlflow models serve -m /opt/jagundi/mlruns/3/49a6f8a2fb30490384136284333ea3c9/artifacts/aspect-extraction -p 1188 -t 180 --env-manager=local
    ```
    test by 
    ```
    curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["sentences"], "data":[["I love this place and menu is great"],["Stay away from this restaurant, the waiter was impatient"],["The price is reasonable but I do not like the flavor of the orange juice"]]}' http://127.0.0.1:1188/invocations

    ```
    
  * run the following cmd to spin up aspect sentiment classification service
    ```
    # bert
    mlflow models serve -m /opt/jagundi/mlruns/3/df2d314dfe074453aeb63779854285fe/artifacts/aspect-sentiment-classification -p 2288 -t 180 --env-manager=local
    
    # distilbert
    mlflow models serve -m /opt/jagundi/mlruns/3/39814bc204594a6b872c57fb3cefdc3d/artifacts/aspect-sentiment-classification -p 2288 -t 180 --env-manager=local
    ```
    test by 
    ```
    curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["sentences", "terms"], "data":[["I love this place and menu is great", "menu"],["Stay away from this restaurant, the waiter was impatient", "waiter"],["The price is reasonable but I do not like the flavor of the orange juice", "price"], ["The price is reasonable but I do not like the flavor of the orange juice", "orange juice"]]}' http://127.0.0.1:2288/invocations
    ```
    
  


