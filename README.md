# CodeClause Data Science Internship Projects

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange.svg)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green.svg)](https://flask.palletsprojects.com)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-yellow.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

This repository contains a comprehensive collection of data science and machine learning projects completed during the CodeClause Data Science Internship. Each project demonstrates different aspects of data science, from computer vision and deep learning to recommendation systems and time series analysis.

## 🚀 Quick Start

**New to this repository?** Check out our [**Setup Guide**](SETUP.md) for step-by-step installation and running instructions!

## 📋 Table of Contents

- [Projects Overview](#-projects-overview)
- [Installation & Setup](#-installation--setup)
- [Project Details](#-project-details)
  - [1. CNNs for Crop Disease Detection](#1-cnns-for-crop-disease-detection)
  - [2. Movie Recommendation System](#2-movie-recommendation-system)
  - [3. K-Means Customer Segmentation](#3-k-means-customer-segmentation)
  - [4. Time Series Food Sales Forecasting](#4-time-series-food-sales-forecasting)
- [Technologies Used](#-technologies-used)
- [Dataset Information](#-dataset-information)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## 🚀 Projects Overview

This repository showcases four distinct data science projects that cover various domains and techniques:

| Project | Domain | Techniques | Application |
|---------|---------|-----------|-------------|
| **Crop Disease Detection** | Computer Vision | CNNs, Deep Learning | Agricultural Technology |
| **Movie Recommendation** | Recommendation Systems | Content-based Filtering | Entertainment |
| **Customer Segmentation** | Business Analytics | K-Means Clustering | Retail Analytics |
| **Time Series Forecasting** | Predictive Analytics | Time Series Analysis | Sales Forecasting |

## 🔧 Installation & Setup

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Required Libraries

Install the required dependencies using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install tensorflow keras opencv-python
pip install flask werkzeug
pip install plotly jupyter notebook
```

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/Sikandarh11/CodeClause-Data-Science-Internship-.git
cd CodeClause-Data-Science-Internship-
```

2. Navigate to any project directory and follow the specific instructions below.

## 📊 Project Details

### 1. CNNs for Crop Disease Detection

**Location:** `CNNs for identifying crop diseases/`

#### 🎯 Objective
Develop a deep learning model to automatically detect and classify potato leaf diseases using Convolutional Neural Networks (CNNs). The project includes a user-friendly web interface for real-time disease detection.

#### 🔬 Technical Implementation
- **Model Architecture:** Custom CNN with data augmentation
- **Dataset:** Potato disease leaf dataset with multiple disease classes
- **Image Processing:** OpenCV for image preprocessing (256x256 pixel resizing)
- **Web Framework:** Flask application with file upload functionality

#### 🏃 How to Run
```bash
cd "CNNs for identifying crop diseases"
python app.py
```
Navigate to `http://localhost:5000` in your browser.

#### 📝 Features
- Upload images of potato leaves
- Real-time disease classification
- Bootstrap-styled responsive web interface
- Support for multiple image formats (PNG, JPG, JPEG)

#### 📊 Model Performance
The CNN model is trained with data augmentation techniques including:
- Rotation, shear, and zoom transformations
- Horizontal flipping
- Target image size: 256x256 pixels
- Batch size: 8

---

### 2. Movie Recommendation System

**Location:** `Movie Recommendation System/`

#### 🎯 Objective
Build a content-based movie recommendation system that suggests movies to users based on movie features and similarity scores.

#### 🔬 Technical Implementation
- **Algorithm:** Content-based filtering
- **Data Processing:** Pandas for data manipulation
- **Similarity Calculation:** Cosine similarity matrices
- **Storage:** Pickle files for model persistence
- **Web Interface:** Flask application with movie search functionality

#### 🏃 How to Run
```bash
cd "Movie Recommendation System"
python app.py
```
Navigate to `http://localhost:5000` for the movie recommendation interface.

#### 📝 Features
- Browse movie catalog
- Get personalized movie recommendations
- Responsive web interface
- Movie metadata display (title, homepage links)

#### 📊 Dataset Features
- Movie titles, genres, and descriptions
- Release dates and ratings
- Homepage URLs and metadata
- Preprocessed similarity scores

---

### 3. K-Means Customer Segmentation

**Location:** `Task_1_(KMeans_Clustering_on_Customer_Segment_Data)/`

#### 🎯 Objective
Perform customer segmentation analysis using K-Means clustering on retail transaction data to identify distinct customer groups for targeted marketing strategies.

#### 🔬 Technical Implementation
- **Clustering Algorithm:** K-Means
- **Data Source:** Online retail transaction data
- **Visualization:** Matplotlib and Seaborn for cluster visualization
- **Analysis:** Customer behavior patterns and purchasing trends

#### 🏃 How to Run
```bash
cd "Task_1_(KMeans_Clustering_on_Customer_Segment_Data)"
jupyter notebook "Task_1_(KMeans_Clustering_on_Customer_Segment_Data).ipynb"
```

#### 📝 Key Insights
- Customer grouping based on purchasing behavior
- Identification of high-value customer segments
- Visual representation of cluster characteristics
- Business recommendations for customer targeting

#### 📊 Analysis Components
- Data preprocessing and cleaning
- Optimal cluster number determination (Elbow method)
- Customer segment profiling
- Statistical analysis of each segment

---

### 4. Time Series Food Sales Forecasting

**Location:** `Time_Series_Forcast_Food_Dataset/`

#### 🎯 Objective
Develop a time series forecasting model to predict food retail sales using historical sales data, enabling better inventory management and business planning.

#### 🔬 Technical Implementation
- **Forecasting Method:** Linear Regression and Time Series Analysis
- **Data:** Monthly food retail sales data (1992 onwards)
- **Visualization:** Matplotlib and Seaborn for trend analysis
- **Evaluation:** Model performance metrics and validation

#### 🏃 How to Run
```bash
cd "Time_Series_Forcast_Food_Dataset"
jupyter notebook "Time_Series_Forcast_Food_Dataset.ipynb"
```

#### 📝 Key Features
- Historical sales trend analysis
- Seasonal pattern identification
- Future sales prediction
- Model accuracy evaluation

#### 📊 Dataset Structure
- **Time Period:** 1992-present
- **Frequency:** Monthly data
- **Features:** Year, Month, Food sales values
- **Format:** CSV file with temporal data

## 🛠️ Technologies Used

### Programming Languages
- **Python 3.7+** - Primary programming language

### Machine Learning & Data Science
- **TensorFlow/Keras** - Deep learning framework
- **Scikit-learn** - Machine learning algorithms
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis

### Visualization
- **Matplotlib** - Statistical plotting
- **Seaborn** - Advanced statistical visualization
- **Plotly** - Interactive visualizations

### Web Development
- **Flask** - Web framework for applications
- **HTML/CSS** - Frontend development
- **Bootstrap** - Responsive web design

### Computer Vision
- **OpenCV** - Image processing and computer vision
- **PIL/Pillow** - Image manipulation

### Data Storage
- **Pickle** - Model serialization
- **CSV** - Data storage format

## 📈 Dataset Information

### Sources and Descriptions

1. **Potato Disease Dataset**
   - Source: Kaggle (potato-disease-leaf-dataset)
   - Format: Image files (PNG/JPG)
   - Classes: Multiple potato disease categories
   - Usage: CNN training and validation

2. **Movie Dataset**
   - Source: The Movies Dataset (Kaggle)
   - Format: CSV files with movie metadata
   - Features: Titles, genres, ratings, descriptions
   - Usage: Content-based recommendation system

3. **Online Retail Dataset**
   - Source: UCI Machine Learning Repository
   - Format: Excel file (.xlsx)
   - Features: Customer transactions, products, quantities
   - Usage: Customer segmentation analysis

4. **Food Retail Sales Dataset**
   - Source: RetailFood.csv (included)
   - Format: CSV file
   - Features: Year, Month, Food sales values
   - Usage: Time series forecasting

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for detailed information on how to contribute to this project.

### Quick Contribution Steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Sikandar** - [GitHub Profile](https://github.com/Sikandarh11)

Project Link: [https://github.com/Sikandarh11/CodeClause-Data-Science-Internship-](https://github.com/Sikandarh11/CodeClause-Data-Science-Internship-)

---

## 🙏 Acknowledgments

- **CodeClause** for providing the internship opportunity
- **Kaggle** for providing datasets
- **UCI Machine Learning Repository** for retail dataset
- **Open source community** for the amazing tools and libraries

---

*This repository represents practical applications of data science concepts ranging from computer vision and deep learning to business analytics and forecasting. Each project is designed to solve real-world problems using appropriate machine learning techniques.*