# Quick Setup Guide

Follow these steps to set up and run the projects in this repository:

## 1. Environment Setup

### Option A: Using pip (Recommended)
```bash
# Clone the repository
git clone https://github.com/Sikandarh11/CodeClause-Data-Science-Internship-.git
cd CodeClause-Data-Science-Internship-

# Install dependencies
pip install -r requirements.txt
```

### Option B: Using conda
```bash
# Create a new conda environment
conda create -n codeclause-ds python=3.8
conda activate codeclause-ds

# Install dependencies
pip install -r requirements.txt
```

## 2. Running Individual Projects

### Crop Disease Detection (Web App)
```bash
cd "CNNs for identifying crop diseases"
python app.py
# Visit http://localhost:5000
```

### Movie Recommendation System (Web App)
```bash
cd "Movie Recommendation System"
python app.py
# Visit http://localhost:5000
```

### Customer Segmentation Analysis (Jupyter Notebook)
```bash
cd "Task_1_(KMeans_Clustering_on_Customer_Segment_Data)"
jupyter notebook "Task_1_(KMeans_Clustering_on_Customer_Segment_Data).ipynb"
```

### Time Series Forecasting (Jupyter Notebook)
```bash
cd "Time_Series_Forcast_Food_Dataset"
jupyter notebook "Time_Series_Forcast_Food_Dataset.ipynb"
```

## 3. Troubleshooting

### Common Issues:

1. **Module not found errors**: Make sure you've installed all requirements
   ```bash
   pip install -r requirements.txt
   ```

2. **Port already in use**: If port 5000 is busy, modify the port in app.py:
   ```python
   app.run(debug=True, port=5001)
   ```

3. **Data files missing**: Some projects require datasets to be downloaded from Kaggle. Check the respective Jupyter notebooks for dataset links.

## 4. Project Structure
```
CodeClause-Data-Science-Internship-/
├── README.md
├── requirements.txt
├── SETUP.md
├── CNNs for identifying crop diseases/
│   ├── app.py
│   ├── CNN_Model.ipynb
│   ├── index.html
│   └── result.html
├── Movie Recommendation System/
│   ├── app.py
│   ├── Movie Recommendation System.ipynb
│   ├── movies.pkl
│   └── templates/
├── Task_1_(KMeans_Clustering_on_Customer_Segment_Data)/
│   └── Task_1_(KMeans_Clustering_on_Customer_Segment_Data).ipynb
└── Time_Series_Forcast_Food_Dataset/
    ├── Time_Series_Forcast_Food_Dataset.ipynb
    └── RetailFood.csv
```

## 5. Next Steps

1. Explore each project's Jupyter notebook to understand the methodology
2. Try the web applications with your own data
3. Modify and experiment with the models
4. Check the main README.md for detailed project descriptions

Happy coding! 🚀