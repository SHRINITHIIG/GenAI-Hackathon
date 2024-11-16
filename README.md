
# Product Recommendation System Using GenAI and LLMs

## Project Overview

This project focuses on building a **personalized product recommendation system** using **Large Language Models (LLMs)** and **GenAI** techniques. The goal is to predict the most suitable product category for a user based on their profile information. The model leverages user demographic data, interests, and purchasing behavior to provide tailored recommendations.

This project implements a **fine-tuned LLM** that classifies users into different product categories, such as **Apparel**, **Electronics**, **Health & Beauty**, and more. The system is deployed using **Gradio** for easy user interaction.

## Dataset

The dataset used in this project contains the following fields:

| Column Name                 | Description |
|-----------------------------|-------------|
| **User_ID**                 | Unique identifier for each user |
| **Age**                     | Age of the user (numeric) |
| **Gender**                  | Gender of the user (Male/Female) |
| **Location**                | User's location (Urban/Suburban/Rural) |
| **Income**                  | User's income (numeric) |
| **Interests**               | The user’s main interest (e.g., Sports, Technology, Fashion, etc.) |
| **Last_Login_Days_Ago**     | Number of days since the last login |
| **Purchase_Frequency**      | Frequency of user purchases (numeric) |
| **Average_Order_Value**     | The average order value (numeric) |
| **Total_Spending**          | Total amount the user has spent on the platform (numeric) |
| **Product_Category_Preference** | Product category the user is most interested in (target label) |
| **Time_Spent_on_Site_Minutes** | Time spent by the user on the website per session (numeric) |
| **Pages_Viewed**            | Number of pages viewed by the user during a session (numeric) |
| **Newsletter_Subscription** | Whether the user has subscribed to the newsletter (True/False) |

The dataset contains **user behavior data**, which will be used to **train** the model to classify the users into their preferred product categories.
Dataset link - https://www.kaggle.com/datasets/kartikeybartwal/ecommerce-product-recommendation-collaborative
## Project Steps

### 1. **Data Preprocessing**
- Cleaned and transformed raw data into a format suitable for training the LLM.
- Applied feature engineering techniques like encoding categorical variables and scaling numerical features.
  
### 2. **Model Training**
- Fine-tuned a pre-trained **BERT** model on the processed dataset for **multi-class classification**.
- Used the model to predict the **Product_Category_Preference** for each user based on their profile data.

### 3. **Evaluation Metrics**
- Evaluated the model using standard metrics such as **Accuracy** and **F1-Score** to ensure the model's predictive quality.
  
### 4. **Model Deployment**
- Deployed the trained model in a **Gradio** interface for easy interaction, allowing users to input their details and receive personalized product recommendations.

### 5. **Recommendation Logic**
- The model predicts the **product category** that the user is most likely to be interested in based on their demographic and behavior data.

## How to Use

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/yourusername/product-recommendation-genai.git
    cd product-recommendation-genai
    ```

2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Launch the Gradio interface:
    ```bash
    python app.py
    ```

4. The Gradio interface will appear in your browser, where you can enter the user's details in the text box in the following format:
    ```plaintext
    Age: 25, Gender: Male, Location: Suburban, Income: 50000, Interests: Sports, Purchase Frequency: 5, Total Spending: 2000
    ```

5. The model will return a **personalized product recommendation** based on the provided user profile.

## Technologies Used

- **Hugging Face Transformers** for fine-tuning BERT.
- **Gradio** for deploying the model as a web interface.
- **PyTorch** for training the model.
- **Scikit-learn** for evaluation metrics (accuracy, F1-score).
- **Pandas** for data manipulation and preprocessing.

## Evaluation

The model was evaluated on the following metrics:
- **Accuracy**: Measures the proportion of correct predictions.
- **F1-Score**: Balances precision and recall for better performance on imbalanced datasets.

## Results

After training the model, we obtained the following evaluation results:
- **Accuracy**: 87%
- **F1-Score**: 0.85

These results demonstrate that the model can accurately classify user profiles into appropriate product categories.

## Future Work

- **Improve model accuracy** by experimenting with different LLM architectures and hyperparameter tuning.
- **Expand the dataset** with more diverse user data to make the recommendations even more personalized.
- **Add more product categories** for a broader range of recommendations.
- **Real-time recommendations**: Incorporate real-time user behavior to improve recommendation accuracy.

## Output
![image](https://github.com/user-attachments/assets/15ea3a2c-6cd9-413c-918f-dedca0c53327)
![image](https://github.com/user-attachments/assets/84172a7c-b0ab-4ce1-b63f-e6625943c8bd)




## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
