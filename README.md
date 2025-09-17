# CodeClause Data Science Internship Portfolio

Welcome to my comprehensive data science internship portfolio repository! This collection showcases four distinct machine learning and data science projects completed during my internship at CodeClause. Each project demonstrates different aspects of data science including computer vision, recommendation systems, clustering analysis, and time series forecasting.

## 📚 Table of Contents

- [Projects Overview](#projects-overview)
- [Technologies Used](#technologies-used)
- [Installation & Setup](#installation--setup)
- [Project Details](#project-details)
  - [1. CNNs for Crop Disease Identification](#1-cnns-for-crop-disease-identification)
  - [2. Movie Recommendation System](#2-movie-recommendation-system)
  - [3. Customer Segmentation using K-Means Clustering](#3-customer-segmentation-using-k-means-clustering)
  - [4. Time Series Forecasting for Food Dataset](#4-time-series-forecasting-for-food-dataset)
- [Repository Structure](#repository-structure)
- [Contact](#contact)

## 🚀 Projects Overview

This repository contains four comprehensive data science projects:

| Project | Domain | Techniques | Deployment |
|---------|--------|------------|------------|
| Crop Disease Detection | Computer Vision | CNN, Deep Learning, Flask | Web Application |
| Movie Recommendation | Recommendation Systems | Collaborative Filtering, Flask | Web Application |
| Customer Segmentation | Clustering & Analytics | K-Means, EDA | Jupyter Notebook |
| Food Sales Forecasting | Time Series Analysis | Regression, Forecasting | Jupyter Notebook |

## 🛠 Technologies Used

- **Programming Languages**: Python
- **Machine Learning**: TensorFlow, Keras, Scikit-learn
- **Data Analysis**: Pandas, NumPy, Matplotlib, Seaborn
- **Web Development**: Flask, HTML, CSS, Bootstrap
- **Deep Learning**: Convolutional Neural Networks (CNNs)
- **Data Processing**: OpenCV, PIL
- **Development Environment**: Jupyter Notebooks, Google Colab

## 📋 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/Sikandarh11/CodeClause-Data-Science-Internship-.git
cd CodeClause-Data-Science-Internship-

# Run the automated setup script (recommended)
chmod +x setup.sh
./setup.sh

# OR install dependencies manually
pip install -r requirements.txt
```

### Manual Installation (Alternative)
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install tensorflow keras opencv-python
pip install flask werkzeug
pip install jupyter notebook
```

### Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Datasets and Models

### Required Datasets
- **Crop Disease Detection**: Potato disease dataset from Kaggle
- **Movie Recommendation**: The Movies Dataset (included as movies.pkl)
- **Customer Segmentation**: Online Retail dataset (Excel format)
- **Time Series Forecasting**: RetailFood.csv (included)

### Pre-trained Models
- `cnn_model_optimized.h5`: Trained CNN model for crop disease detection
- `movies.pkl`: Processed movie dataset with features
- `similarity_scores.pkl`: Precomputed similarity matrix for recommendations
- `pt.pkl`: Additional processed data for movie recommendations

### Data Sources
- **Kaggle**: Primary source for crop disease and retail datasets
- **Movie Database**: Comprehensive movie metadata
- **Retail Data**: Customer transaction records for segmentation analysis

## 🔍 Project Details

### 1. CNNs for Crop Disease Identification

A computer vision application that uses Convolutional Neural Networks to identify diseases in crop images, specifically focusing on potato leaf diseases.

#### Features
- **Real-time Disease Detection**: Upload plant images and get instant disease classification
- **Web Interface**: User-friendly Flask web application
- **Deep Learning Model**: Custom CNN architecture trained on potato disease dataset
- **High Accuracy**: Achieved 83.7% accuracy on test dataset

#### Technologies
- TensorFlow/Keras for deep learning
- OpenCV for image processing
- Flask for web deployment
- Bootstrap for responsive UI

#### How to Run
```bash
cd "CNNs for identifying crop diseases"
python app.py
```
Navigate to `http://localhost:5000` in your browser.

#### Model Performance
- **Training Accuracy**: ~90%
- **Test Accuracy**: 83.7%
- **Classes**: Healthy, Early Blight, Late Blight

### 2. Movie Recommendation System

An intelligent movie recommendation system that suggests movies based on user preferences using collaborative filtering techniques.

#### Features
- **Personalized Recommendations**: Get movie suggestions based on your interests
- **Web Interface**: Interactive Flask application
- **Large Dataset**: Built using comprehensive movie metadata
- **Real-time Processing**: Instant recommendations upon input

#### Technologies
- Scikit-learn for recommendation algorithms
- Pandas for data manipulation
- Flask for web deployment
- Bootstrap for styling

#### How to Run
```bash
cd "Movie Recommendation System"
python app.py
```
Navigate to `http://localhost:5000` in your browser.

#### Dataset
- Based on comprehensive movie database
- Features include: title, genres, ratings, popularity
- Preprocessed similarity scores for efficient recommendations

### 3. Customer Segmentation using K-Means Clustering

A comprehensive analysis project that segments customers based on their purchasing behavior using unsupervised machine learning techniques.

#### Features
- **Customer Segmentation**: Identify distinct customer groups
- **Behavioral Analysis**: Understand purchasing patterns
- **Data Visualization**: Comprehensive charts and plots
- **Business Insights**: Actionable recommendations for marketing strategies

#### Technologies
- Scikit-learn for K-Means clustering
- Pandas and NumPy for data processing
- Matplotlib and Seaborn for visualization
- Statistical analysis techniques

#### How to Run
```bash
cd "Task_1_(KMeans_Clustering_on_Customer_Segment_Data)"
jupyter notebook "Task_1_(KMeans_Clustering_on_Customer_Segment_Data).ipynb"
```

#### Key Insights
- Optimal number of clusters determined using elbow method
- Customer segments based on RFM analysis (Recency, Frequency, Monetary)
- Visualization of customer distribution and characteristics

### 4. Time Series Forecasting for Food Dataset

A predictive analytics project that forecasts food sales using time series analysis and regression techniques.

#### Features
- **Sales Forecasting**: Predict future food sales trends
- **Trend Analysis**: Identify seasonal patterns and trends
- **Statistical Modeling**: Multiple forecasting approaches
- **Data Visualization**: Comprehensive trend analysis charts

#### Technologies
- Scikit-learn for regression models
- Pandas for time series manipulation
- Matplotlib for trend visualization
- Statistical forecasting methods

#### How to Run
```bash
cd "Time_Series_Forcast_Food_Dataset"
jupyter notebook "Time_Series_Forcast_Food_Dataset.ipynb"
```

#### Key Features
- Historical data analysis
- Trend identification and seasonal decomposition
- Multiple forecasting models comparison
- Performance metrics evaluation

## 📁 Repository Structure

```
CodeClause-Data-Science-Internship-/
│
├── CNNs for identifying crop diseases/
│   ├── app.py                          # Flask web application
│   ├── CNN_Model.ipynb                 # Model training notebook
│   ├── cnn_model_optimized.h5          # Trained model file
│   ├── index.html                      # Main page template
│   └── result.html                     # Results page template
│
├── Movie Recommendation System/
│   ├── app.py                          # Flask web application
│   ├── Movie Recommendation System.ipynb # Analysis notebook
│   ├── movies.pkl                      # Movie dataset
│   ├── similarity_scores.pkl           # Precomputed similarities
│   ├── pt.pkl                         # Additional data
│   ├── index.html                      # Main page template
│   └── recommend.html                  # Recommendation page
│
├── Task_1_(KMeans_Clustering_on_Customer_Segment_Data)/
│   └── Task_1_(KMeans_Clustering_on_Customer_Segment_Data).ipynb
│
├── Time_Series_Forcast_Food_Dataset/
│   ├── Time_Series_Forcast_Food_Dataset.ipynb
│   └── RetailFood.csv                  # Dataset
│
└── README.md                           # This file
```

## 🎯 Learning Outcomes

Through these projects, I have gained expertise in:

- **Deep Learning**: Implementing CNNs for image classification
- **Web Development**: Creating Flask applications for ML deployment
- **Data Analysis**: Exploratory data analysis and statistical modeling
- **Machine Learning**: Supervised and unsupervised learning techniques
- **Data Visualization**: Creating meaningful insights through charts and plots
- **Time Series Analysis**: Forecasting and trend analysis
- **Recommendation Systems**: Building collaborative filtering systems

## 🔧 Troubleshooting

### Common Issues

#### Flask Applications
- **Port Already in Use**: Change the port in `app.py`: `app.run(debug=True, port=5001)`
- **Module Not Found**: Ensure all dependencies are installed: `pip install -r requirements.txt`
- **File Permissions**: Make sure the uploads folder has write permissions

#### Jupyter Notebooks
- **Kernel Issues**: Restart kernel and run all cells
- **Missing Dependencies**: Install required packages in the notebook environment
- **Dataset Not Found**: Ensure datasets are in the correct directories

#### Model Loading
- **TensorFlow Version**: Ensure TensorFlow 2.x is installed
- **Model File Corruption**: Re-download the model file if needed
- **Memory Issues**: Use CPU instead of GPU if memory is limited

### Performance Tips
- Use virtual environments to avoid dependency conflicts
- For faster training, use GPU if available
- Reduce batch size if encountering memory issues
- Use smaller datasets for testing purposes

## 🚀 Future Enhancements

- **Crop Disease Detection**: Expand to more crop types and diseases
- **Movie Recommendation**: Implement deep learning-based recommendations
- **Customer Segmentation**: Add real-time clustering capabilities
- **Time Series**: Implement advanced forecasting models (ARIMA, LSTM)

## 📞 Contact

**Sikandar Hassan**
- GitHub: [@Sikandarh11](https://github.com/Sikandarh11)
- Project Repository: [CodeClause-Data-Science-Internship-](https://github.com/Sikandarh11/CodeClause-Data-Science-Internship-)

---

**Note**: This repository represents my learning journey and practical application of data science concepts during my internship at CodeClause. Each project is designed to showcase different aspects of the data science pipeline from data collection to model deployment.

## 📜 License

This project is open source and available under the [MIT License](LICENSE).