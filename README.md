# **Mobile Price Range Prediction**
  
![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=matplotlib&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-43B02A?style=for-the-badge&logo=seaborn&logoColor=white) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)  
**Other Tools**: NumPy, Jupyter Notebooks

---

## **Overview**

![Mobile Image](https://drive.google.com/uc?export=view&id=1dOcrlRovlyOfAcB2bbpMGChMKPouDlE7)

This project predicts the price range of mobile phones based on their features using classification algorithms. We analyzed key specifications like **RAM**, **battery power**, and **camera quality** to classify phones into one of four price categories: **Low**, **Medium**, **High**, and **Very High**.

---

## **Problem Statement**

With mobile phone features evolving rapidly, predicting the price category based on specifications is essential for competitive pricing. This project uses machine learning to solve this classification problem, providing businesses with insights to strategically price their products.

---

## **Dataset Description**

- **[Download Dataset](https://drive.google.com/file/d/1tbaIkP79hq9wagJ3w4ZZMilOtzV48twy/view?usp=sharing)**  
The dataset contains 2000 records of mobile phone specifications:
  - **RAM**: Memory capacity of the phone
  - **Battery Power**: Battery capacity in mAh
  - **Pixel Resolution**: Screen resolution in pixels
  - **Camera Specs**: Front and rear camera megapixels
  - **Additional Features**: Bluetooth, 4G, etc.

---

## **Analysis Approach**

1. **Data Preprocessing**:  
   - No missing values.
   - Scaled numerical features for better model performance.

2. **Exploratory Data Analysis (EDA)**:  
   - Visualized feature distributions and identified important variables affecting price range.
   - Analyzed the correlation between features using heatmaps.

3. **Model Development**:  
   - Applied classification algorithms: **Logistic Regression**, **SVM**, **KNN**, **Random Forest**, **XGBoost**, and **Gradient Boosting Classifier**.
   - Evaluated model performance using **accuracy**, **precision**, **recall**, and **ROC AUC**.

---

## **Key Insights**

### 1. Feature Importance
- **RAM** emerged as the most important factor influencing the price range.
- **Battery Power** and **Pixel Resolution** were also critical features for classification.

### 2. RAM Distribution Across Price Ranges
Higher RAM significantly increases the likelihood of a phone being categorized in the higher price range.

### 3. Camera Quality Impact
While **camera specifications** contribute to the classification, they have a relatively smaller impact compared to RAM and battery power.

---

## **Model Performance**

| **Model**              | **Accuracy** | **Precision** | **Recall** | **ROC AUC** |
|------------------------|--------------|---------------|------------|-------------|
| **Logistic Regression** | 96.43%       | 96.45%        | 96.43%     | 99.75%      |
| **SVM**                | 96.15%       | 96.16%        | 96.15%     | 99.87%      |
| **KNN**                | 91.76%       | 91.76%        | 91.76%     | 97.75%      |
| **XGBoost**            | 90.66%       | 90.75%        | 90.66%     | 99.09%      |
| **Random Forest**       | 89.56%       | 89.58%        | 89.56%     | 98.88%      |
| **Gradient Boosting**   | 89.28%       | 89.24%        | 89.28%     | 98.79%      |

### Best Model: **Logistic Regression**  
The Logistic Regression model provided the highest accuracy of **96.43%** with excellent precision and recall scores.

---

## **Challenges & Limitations**

1. **Feature Engineering**: While the dataset included critical phone features, additional features (e.g., brand, user reviews) could improve model accuracy.
2. **Limited Scope**: The model is trained on a relatively small dataset and may not generalize well to other datasets.
3. **Hyperparameter Tuning**: Optimizing hyperparameters for models like **SVM** and **XGBoost** could further improve performance.

---

## **Future Scope**

1. **Feature Expansion**: Including additional factors like **brand** or **customer reviews** could provide deeper insights.
2. **Predictive Pricing**: Using regression models to predict the exact price instead of classifying into price ranges.
3. **Model Deployment**: Implementing a web app for real-time mobile price predictions based on user inputs.

---

## **Resources**

- **IPYNB Notebook**: [Colab Link](https://colab.research.google.com/drive/1XzGN-l_XBfdbSexgw3hG-bX5RQuR0vFg?usp=sharing)  
- **GitHub Repository**: [GitHub Link](https://github.com/Rudrajit12/Mobile-Price-Range-Prediction)

---

## **References**

1. [Kaggle - Mobile Price Classification](https://www.kaggle.com/iabhishekofficial/mobile-price-classification)
2. [Towards Data Science - Classification Models](https://towardsdatascience.com/classification-models-overview-5a5f7e54d9e1)

---

## **About the Author**

**Author**: Rudrajit Bhattacharyya  
This project was developed as part of the **AlmaBetter Full Stack Data Science** program.

- **Email**: [rudrajitb24@gmail.com](mailto:rudrajitb24@gmail.com)  
- **LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/rudrajitb/)  
- **GitHub**: [GitHub Profile](https://github.com/Rudrajit12)

---
