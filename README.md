# **Mobile Price Range Prediction**

![Project Category](https://img.shields.io/badge/Project%20Category-Classification-blue) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Enabled-brightgreen)

**Tools & Technologies Used**:  
![Python](https://img.shields.io/badge/Python-3.9-blue) ![NumPy](https://img.shields.io/badge/NumPy-Enabled-orange) ![Pandas](https://img.shields.io/badge/Pandas-Enabled-yellowgreen) ![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-red) ![Seaborn](https://img.shields.io/badge/Seaborn-Data%20Visualization-yellow) ![Jupyter Notebooks](https://img.shields.io/badge/Jupyter-Notebook-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-green)

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

## **Impact & Implications**

The project's findings are like treasure for phone companies! They help these companies make smart decisions about how much to charge for their phones. Here's how it helps:

1. **Being the Best Price**: Companies can figure out exactly how much their phones should cost, so they can compete well with other companies. They can make sure their prices are just right to attract customers and still make money.
2. **Finding the Right Spot**: Knowing what features make phones more expensive helps companies decide where their phones fit best in the market. They can aim their phones at people who want fancy stuff or those looking for a good deal.
3. **Standing Out**: By knowing what makes their phones pricier, companies can make them special. They can show off cool features or let people customize their phones, making them different from others in the market.
4. **Getting the Word Out**: Understanding what prices people are willing to pay and what they like helps companies advertise better. They can make ads that speak to different kinds of customers, making their ads more effective.
5. **Making Better Choices**: With all this smart info, the big bosses at these companies can make better decisions about everything from making new phones to how much to spend on ads. It helps them use their time and money in the best way possible, leading to more success for their business.

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
