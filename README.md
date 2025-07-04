# 🍿 Netflix Review Sentiment Analysis

![GitHub stars](https://img.shields.io/github/stars/Sivaprasad-creator/Netflix-Review-Analysis)
![GitHub forks](https://img.shields.io/github/forks/Sivaprasad-creator/Netflix-Review-Analysis)
![GitHub license](https://img.shields.io/github/license/Sivaprasad-creator/Netflix-Review-Analysis)

![Netflix Review](https://github.com/Sivaprasad-creator/Netflix-Review-Analysis/blob/main/netflix_banner.jpg)

---

## 📌 Project Title

**Netflix Review Sentiment Analysis**

---

## 📝 Project Overview

This project focuses on performing **sentiment analysis** on Netflix app reviews collected from the Google Play Store. Using natural language processing and machine learning techniques, the model predicts whether a review is **Positive**, **Negative**, or **Neutral** based on the review content.

> 🔧 **Tools Used:** Python, NLP, Machine Learning, TF-IDF, Streamlit

---

## 📁 Dataset Information

- **Source:** [Kaggle - Netflix Play Store Reviews](https://www.kaggle.com/datasets/ashishkumarak/netflix-reviews-playstore-daily-updated/data)
- **Sample Size:** 138,947 reviews
- **Key Columns:**
  - `reviewId` — Unique identifier of the review
  - `username` — Name of the reviewer
  - `content` — Review text
  - `score` — Rating given (1 to 5 stars)
  - `thumbsUpCount` — Number of likes for the review
  - `reviewCreatedVersion` — App version at the time of review
  - `at` — Timestamp of the review

---

## 🎯 Objectives

- Clean and preprocess user review text
- Handle imbalanced sentiment classes effectively
- Vectorize reviews using TF-IDF with n-grams
- Build and compare multiple classification models
- Deploy the best model using a Streamlit web app for real-time prediction

---

## 📊 Analysis Summary

- **Data Cleaning:** Removed nulls, special characters, and redundant spaces
- **Preprocessing:** Tokenization, Lemmatization, Stopword removal
- **EDA:** Word clouds, rating distributions, review length analysis
- **TF-IDF:** Applied with unigrams and bigrams for better context
- **Balancing:** Used RandomOverSampler to tackle class imbalance
- **Model Evaluation:** Used accuracy, classification report, confusion matrix, and ROC curve for evaluation
- **Deployment:** Final model deployed with Streamlit

---

## 🤖 Models Used

| Stage              | Libraries/Techniques Used                                                                                                      |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------|
| **Preprocessing**  | `pandas`, `numpy`, `re`, `nltk`, `stopwords`, `lemmatizer`, `seaborn`, `matplotlib`, `wordcloud`                              |
| **Feature Extraction** | `TfidfVectorizer` with n-grams                                                                                          |
| **Class Balancing**| `RandomOverSampler` from `imblearn`                                                                                           |
| **Models**         | `LogisticRegression`, `RidgeClassifier`, `MultinomialNB`, `LinearSVC`, `DecisionTreeClassifier`, `RandomForestClassifier`     |
| **Ensemble**       | `VotingClassifier`, `StackingClassifier`                                                                                      |
| **Metrics**        | `accuracy_score`, `classification_report`, `confusion_matrix`, `roc_auc_score`, `roc_curve`                                  |
| **Model Saving**   | `joblib`                                                                                                                       |
| **Deployment**     | **Streamlit**                                                                                                                  |

---

## 🚀 Streamlit Deployment

This project includes a **Streamlit app** for real-time sentiment prediction:

- Input review text in the app
- Model predicts if the review is **Positive**, **Negative**, or **Neutral**
- Displays probabilities and interpretation

---

## 🛠️ How to Run Locally

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Sivaprasad-creator/Netflix-Review-Analysis.git
   cd Netflix-Review-Analysis
   ```

2. **Create and Activate Virtual Environment (optional but recommended)**
   - On Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```

---

## 📦 Requirements

Below are the required packages listed in `requirements.txt`:

```txt
pandas
numpy
matplotlib
seaborn
nltk
scikit-learn
joblib
wordcloud
imblearn
streamlit
```

---

## 👨‍💻 Author Info

**Sivaprasad T.R**  
📧 Email: sivaprasadtrwork@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/sivaprasad-t-r)  
💻 [GitHub](https://github.com/Sivaprasad-creator)

---

## 📜 Data Source

Data sourced from: [Netflix Reviews on Kaggle](https://www.kaggle.com/datasets/ashishkumarak/netflix-reviews-playstore-daily-updated/data)

---

## 🙏 Acknowledgements

Thanks to Kaggle and the original dataset contributors.

---

## 💬 Feedback

Feel free to open issues or connect on LinkedIn/GitHub for improvements or collaborations.
