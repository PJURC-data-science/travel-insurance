![banner](https://github.com/PJURC-data-science/travel-insurance/blob/main/media/banner.png)

# Travel Insurance Prediction: Customer Purchase Analysis
[View Notebook](https://github.com/PJURC-data-science/travel-insurance/blob/main/Travel%20Insurance.ipynb)

A machine learning analysis to predict customer likelihood of purchasing health-related travel insurance. This study develops a predictive model using customer demographics and travel history to identify potential insurance buyers.

## Overview

### Business Question 
How can we predict which customers are most likely to purchase travel insurance using insights from COVID-19 era data?

### Key Findings
- Age and family size strongly influence purchases
- Frequent flyer status is significant predictor
- International travel history impacts decisions
- Income level correlates with purchase likelihood
- Model achieves 82% accuracy rate

### Impact/Results
- Developed predictive model
- Identified key customer segments
- Quantified feature importance
- Established prediction benchmarks
- Created targeting framework

## Data

### Source Information
- Dataset: Travel Insurance Prediction Data
- Source: Kaggle Dataset
- Size: ~2000 customer records
- Context: Indian travel market, COVID-19 period

### Variables Analyzed
- Customer demographics
- Travel history
- Family size
- Frequent flyer status
- Income levels
- Insurance purchase decisions

## Methods

### Analysis Approach
1. Exploratory Analysis
   - Distribution analysis
   - Feature correlation
   - Significance testing
2. Feature Engineering
   - Variable selection
   - Demographic analysis
   - Travel pattern analysis
3. Model Development
   - Logistic Regression
   - Stacking Classifier
   - Performance optimization

### Tools Used
- Python (Data Science)
  - Pandas: Data manipulation
  - Scikit-learn:
    - Logistic Regression
    - Stacking Classifier
    - Train-test splitting
    - Grid Search CV
    - Performance metrics
    - ROC curve analysis
  - Matplotlib/Seaborn: Visualization
  - Scipy: Statistical testing
  - Performance Metrics:
    - Accuracy (82%)
    - Confusion Matrix
    - ROC Curve
    - Precision-Recall Curve

## Getting Started

### Prerequisites
```python
imbalanced_learn==0.12.3
matplotlib==3.8.4
numpy==2.2.0
pandas==2.2.3
scikit_learn==1.6.0
seaborn==0.13.2
statsmodels==0.14.4
```

### Installation & Usage
```bash
git clone git@github.com:PJURC-data-science/travel-insurance.git
cd travel-insurance
pip install -r requirements.txt
jupyter notebook "Travel Insurance.ipynb"
```

## Project Structure
```
travel-insurance/
│   README.md
│   requirements.txt
│   Travel Insurance.ipynb
|   utils.py
└── data/
    └── TravelInsurancePrediction.csv
```

## Strategic Recommendations
1. **Customer Targeting**
   - Focus on key demographics
   - Leverage travel history
   - Consider family size
   - Account for income levels

2. **Model Application**
   - Implement predictive scoring
   - Monitor performance
   - Adjust for US market

3. **Performance Optimization**
   - Improve recall rates
   - Balance precision-recall
   - Reduce false negatives
   - Enhance prediction accuracy

## Future Improvements
- Expand dataset size
- Address cultural differences
- Enhance recall metrics
- Expand grid search
- Add feature engineering
- Test more advanced models (Gradient Boosting, Neural Network, etc.)
- Test additional ensemble methods (e.g., VotingClassifier with weights)
- Collect US market data