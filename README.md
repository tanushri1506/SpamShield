# SpamShield: Spam Detection System using NLP & Machine Learning

## Project Overview
This project builds an end-to-end **Spam Detection System** using Natural Language Processing (NLP) and Machine Learning to classify SMS messages as **Spam** or **Ham**. The workflow covers data cleaning, exploratory data analysis (EDA), text preprocessing, feature extraction, model comparison, and deployment-ready model serialization.

The dataset is **highly imbalanced**, so special emphasis is placed on **precision** to minimize false positives.

## Dataset
- **Source:** SMS Spam Collection Dataset
- **Records:** 5,168 unique SMS messages (after cleaning)
- **Target Variable:**  
  - `0` → Ham  
  - `1` → Spam

### Key Columns
- `Message`: Raw SMS text  
- `Category`: Spam / Ham label  

## Tools & Technologies
- **Programming Language:** Python  
- **Libraries:**  
  - Data Analysis: Pandas, NumPy  
  - NLP: NLTK  
  - Visualization: Matplotlib, Seaborn, WordCloud  
  - Machine Learning: Scikit-learn, XGBoost  
- **Environment:** Jupyter Notebook  

## Project Workflow

### 1. Data Cleaning
- Removed irrelevant columns and encoding issues
- Renamed columns for clarity
- Encoded target labels using `LabelEncoder`
- Removed duplicate messages (404 duplicates)
- Final dataset size: **5,168 × 2**

### 2. Exploratory Data Analysis (EDA)
- Analyzed class distribution (**~87% Ham, ~13% Spam**)
- Identified strong **class imbalance**
- Extracted:
  - Number of characters
  - Number of words
  - Number of sentences
- Compared linguistic patterns between spam and ham messages
- Visualizations:
  - Pie charts
  - Histograms
  - Pair plots
  - Correlation heatmaps

### 3. Text Preprocessing
- Converted text to lowercase
- Tokenization using NLTK
- Removed:
  - Stopwords
  - Punctuation
  - Non-alphanumeric tokens
- Applied **Porter Stemming**
- Created a cleaned `Transformed_Message` column

### 4. Feature Engineering
- Generated word clouds for spam and ham messages
- Identified most frequent words in spam vs ham
- Converted text into numerical features using:
  - **TF-IDF Vectorization** (`max_features = 3000`)

### 5. Model Building
Trained and evaluated multiple classification models:

- Naive Bayes (Gaussian, Multinomial, Bernoulli)
- Logistic Regression
- Support Vector Classifier (SVC)
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- AdaBoost
- Extra Trees
- Gradient Boosting
- XGBoost

### 6. Model Evaluation
- Evaluation metrics:
  - **Accuracy**
  - **Precision** (primary metric due to imbalance)

#### Key Results
- **Multinomial Naive Bayes (TF-IDF)**  
  - Accuracy: **97.0%**
  - Precision: **1.00**

- Ensemble Models:
  - Voting Classifier: **98.1% accuracy, 1.00 precision**
  - Stacking Classifier: **98.1% accuracy, 1.00 precision**

Despite strong ensemble performance, **Multinomial Naive Bayes** was selected due to:
- High precision
- Simplicity
- Faster inference
- Better deployment suitability

### 7. Model Improvement
- Tuned TF-IDF `max_features`
- Compared scaling and additional numeric features
- Verified consistent performance across configurations

### 8. Deployment Preparation
- Serialized trained components using `pickle`:
  - `vectorizer.pkl`
  - `model.pkl`
- These files can be directly used in a **Streamlit / Flask web app** for real-time spam detection.

## Key Insights
- Spam messages are significantly **longer** and more **information-dense**
- Pricing, reward, and urgency-related keywords dominate spam corpus
- Precision-focused modeling is crucial for imbalanced NLP problems
- Simple probabilistic models can outperform complex models for text classification

## Conclusion
This project demonstrates a complete NLP pipeline—from raw text to a deployment-ready model—highlighting the importance of preprocessing, feature engineering, and metric selection. The final system achieves **high precision**, making it suitable for real-world spam filtering applications.

## Contact
- **Name:** Tanushri Barsainya  
- **GitHub:** https://github.com/tanushri1506  
- **LinkedIn:** https://www.linkedin.com/in/tanushri1506/
