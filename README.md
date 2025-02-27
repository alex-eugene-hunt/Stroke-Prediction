# Stroke Prediction Model

## Project Overview
This project implements machine learning models to predict the likelihood of stroke occurrence based on various health indicators and demographic factors. The implementation focuses on comparing different algorithms and their effectiveness in medical prediction tasks.

## Technical Implementation

### Machine Learning Algorithms
1. **Random Forests**
   - Custom implementation of Random Forest classifier
   - Features:
     - Bootstrap sampling for bagging
     - Information gain calculation using entropy
     - Majority voting for ensemble predictions
     - Hyperparameter tuning capabilities
   - Configurable parameters:
     - Number of trees (n_trees)
     - Maximum depth (max_depth)
     - Feature selection methods (sqrt, log2)

2. **Decision Trees**
   - Custom implementation with entropy-based splitting
   - Recursive tree building with configurable depth
   - Information gain optimization

![Screenshot 2025-02-27 140403](https://github.com/user-attachments/assets/b2df591f-4cb0-4488-9ae2-8f364d220237)
![Screenshot 2025-02-27 140441](https://github.com/user-attachments/assets/fbb47e76-6ef4-46c7-9aab-5388294efe28)


3. **Novelty Detection**
   - Implementation for anomaly detection in medical data
   - Specialized approach for imbalanced medical datasets

### Data Processing
- **Feature Engineering**
  - One-hot encoding for categorical variables
  - Missing value imputation for BMI using mean strategy
  - Numerical feature scaling

- **Dataset**
  - Healthcare stroke dataset with multiple features
  - Handles various data types:
    - Categorical: gender, marriage status, work type, residence type, smoking status
    - Numerical: age, BMI, glucose level
    - Binary: hypertension, heart disease

### Technical Stack
- **Programming Language**: Python
- **Key Libraries**:
  - `numpy`: Numerical computations and array operations
  - `pandas`: Data manipulation and analysis
  - `matplotlib`: Data visualization and model performance plots

### Model Evaluation
- Confusion matrix implementation
- Accuracy calculation
- Cross-validation support
- Grid search for hyperparameter optimization

### Code Structure
```
AlexHunt_SL/
├── AlexHunt_Random-Forests.py  # Random Forest implementation
├── AlexHunt_Decision-Trees.py  # Decision Tree implementation
├── AlexHunt_Novelty.py        # Novelty detection implementation
└── healthcare-dataset-stroke-data.csv  # Dataset
```

## Technical Highlights
- Custom implementation of machine learning algorithms from scratch
- Emphasis on algorithm optimization and performance
- Robust data preprocessing pipeline
- Comprehensive model evaluation framework
- Focus on interpretability in medical context

## Future Enhancements
- Implementation of additional ensemble methods
- Feature importance analysis
- ROC curve analysis
- Cross-validation optimization
- Model deployment pipeline

## Skills Demonstrated
- Advanced Python programming
- Machine learning algorithm implementation
- Data preprocessing and feature engineering
- Statistical analysis and model evaluation
- Medical data handling
- Algorithm optimization
- Scientific computing with NumPy and Pandas
