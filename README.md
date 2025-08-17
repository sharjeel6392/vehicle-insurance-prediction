# Vehicle Insurance Prediction Project - An end-to-end MLOps Pipeline from scratch.
This project implements a complete MLOps pipeline to predict whether a customer is likely to buy vehicle insurance or not. The pipeline covers data ingestion, validation, transformation, model training, evaluation, and deployment.

## Project Flow
This project follows a structured MLOps pipeline. The key steps are:

1. **Project Setup:** Initializing the project template, configuring setup.py and pyproject.toml for local package imports, and setting up the virtual environment with required dependencies.

2. **MongoDB Setup:** Configuring *MongoDB Atlas* for data storage, including user creation, network access, and obtaining connection strings. Data is then pushed to MongoDB from a Python notebook.

3. **Logging and Exception Handling:** Implementing robust logging and exception handling mechanisms. *Exploratory Data Analysis (EDA)* and *Feature Engineering* are performed in Jupyter notebooks.

4. **Data Ingestion:** Defining constants, configuring MongoDB connections, creating data access modules to fetch and transform data into DataFrames, and implementing the data ingestion component within the training pipeline.

5. **Data Validation, Transformation, and Model Training:** Completing data validation using a schema, performing data transformation, and training the machine learning model. This includes defining estimator classes.

6. **AWS Setup for Model Evaluation and Pusher:** Configuring *AWS IAM user* with administrative access, setting up *S3* bucket for model storage, and implementing S3 utility functions for model pull/push operations.

7. **Model Evaluation and Model Pusher:** Implementing the model evaluation and model pusher components, leveraging AWS S3 for model registry.

8. **Prediction Pipeline and Web Application:** Structuring the prediction pipeline and developing a *web application* using app.py with static and templates directories for frontend.

9. **CI/CD Process with GitHub Actions and AWS EC2/ECR:** Setting up *Dockerfile* and .dockerignore, configuring *GitHub Actions* for *CI/CD*, creating AWS IAM user for CI/CD, setting up *ECR* for *Docker image* storage, and deploying the application on *AWS EC2* with Docker.

## Data

The project utilizes a dataset located at `notebook/data.csv` within the repository. This dataset is used for training and evaluating the vehicle insurance prediction model. A detailed breakdown of the columns and their descriptions is as follows:

* **id:** Unique ID for the customer
* **Gender:** Gender of the customer
* **Age:** Age of the customer
* **Driving_License:** Whether a customer has a driver's license. 0: No. 1: Yes
* **Region_Code:** Unique code for the region of the customer.
* **Previously_insured:** Whether the customer already has a verhicle insurance. 0: No. 1: Yes
* **Vehicle_Age:** Age of the vehicle
* **Vehicle_Damage:** Whether the vehicle has a damage history. 0: No. 1: Yes
* **Annual_Premium:** The amount a customer needs to pay as premium in a year
* **Policy_Sales_Channel:** Anonymized code for the channel of outreaching to customers. i.e, Agents, Mail, Phone, in person, etc.
* **Vintage:** Number of days that customer has been associated with the company
* **Response:** Whether the customer is interested in buying the insurance. 0: No. 1: Yes

## Random Forest Classifier

The core of this project is a machine learning model designed to predict whether a customer is likely to buy vehicle insurance. This is a classification problem. The model used is a `RandomForestClassifier`, with hyperparameters chosen after extensive tuning using `RandomizedSearchCV`. The optimized parameters are:

* n_estimators: 200
* min_samples_split: 7
* min_samples_leaf: 6
* max_depth: 10
* criterion: "entropy"
* random_state: 101

The specific details of the model architecture, training parameters, and evaluation metrics are defined within the `src/entity/estimator.py` file. The model is trained as part of the MLOps pipeline, ensuring that it is continuously updated and optimized with new data.




## Installation and Setup

To set up the project locally, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/sharjeel6392/vehicle-insurance-prediction.git
    cd vehicle-insurance-prediction
    ```
2.  **Create and activate a virtual environment**:
    ```bash
    conda create -n proj1 python=3.10 -y
    conda activate proj1
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **MongoDB Configuration**: To set up MongoDB Atlas and obtain your connection string, please refer to the project\'s documentation or relevant setup guides. Set the `MONGODB_URL` environment variable:
    ```bash
    # For Windows PowerShell
    $env:MONGODB_URL = "mongodb+srv://<username>:<password>@cluster0.qvdpoh5.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

    # For Linux/macOS Bash
    export MONGODB_URL="mongodb+srv://<username>:<password>@cluster0.qvdpoh5.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    ```
    Replace `<username>` and `<password>` with your MongoDB Atlas credentials.

5.  **AWS Configuration (for deployment)**: To set up AWS IAM user, S3 bucket, configure environment variables for `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` and setup EC2 environment. Furthermore, install docker on the EC2 OS (typically Ubuntu server), and create a Github Actions runner module, linking it to AWS EC2.

## CI/CD Pipeline

The project includes a CI/CD pipeline configured with GitHub Actions and AWS EC2/ECR. Upon every push to the repository, the pipeline automatically builds a Docker image, pushes it to AWS ECR, and deploys the updated application to the EC2 instance.

## Usage
To run the application locally:
1.  Ensure all installation and setup steps are completed.
2.  Run the `app.py` file:
    ```bash
    python app.py
    ```
3.  Access the web application in your browser at `http://localhost:5000` (or the port specified by Flask).

To run the application from anywhere:

1.  Ensure all installation and setup steps are completed.
2.  Execute the command `./run.sh/` on EC2, and grab the URL of the said EC2 from AWS console.
3.  Access the web application in your browser at the EC2 ip address:port exposed.

## Future Work

This project provides a solid foundation for an MLOps pipeline. Here are some potential areas for future enhancements:

*   **Advanced Model Monitoring**: Implement more sophisticated model monitoring techniques to detect data drift, concept drift, and model performance degradation in real-time. This could involve integrating with tools like Evidently AI or MLflow.
*   **Automated Retraining**: Develop a mechanism for automated model retraining based on performance metrics or data changes. This would ensure the model remains accurate and relevant over time.
*   **A/B Testing for Models**: Integrate A/B testing capabilities to compare the performance of different model versions in a production environment before full deployment.
*   **Scalability and High Availability**: Enhance the deployment infrastructure to support higher traffic loads and ensure high availability of the prediction service. This might involve using container orchestration (e.g., Kubernetes) and load balancing.
*   **Security Enhancements**: Implement more robust security measures, including fine-grained access control for AWS resources, secure API endpoints, and data encryption at rest and in transit.
*   **Cost Optimization**: Explore strategies to optimize the cost of cloud resources used in the MLOps pipeline, such as using spot instances for training or optimizing storage.
*   **User Interface Improvements**: Develop a more interactive and user-friendly web interface for the prediction service, potentially including data visualization and input forms.
*   **Expand Data Sources**: Integrate additional data sources to enrich the dataset and potentially improve model accuracy.
*   **Experiment Tracking**: Implement a dedicated experiment tracking system (e.g., MLflow, Weights & Biases) to manage and compare different model training runs and their artifacts.

## Contributing

Contributions are welcome! Please feel free to fork the repository, create a new branch, and submit pull requests. Ensure your code adheres to the project\'s coding standards and includes appropriate tests.

## Contact

For any questions or inquiries, please contact sharjeel6392 (GitHub username).