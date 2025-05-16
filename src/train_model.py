import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from data_preprocessing import preprocess_data
from model_pipeline import get_model_pipeline

from scipy.stats import randint, uniform

def train_and_tune(df, model_name='random_forest', n_iter=50):
    df = preprocess_data(df)

    # Define features and target
    numerical_features = [
        'habitableSurface', 'bedroomCount', 'bathroomCount', 'facadeCount',
        'landSurface', 'buildingConstructionYear', 'epc_enum', 'floodZoneType_mapping',
        'buildingCondition_mapping', 'postcode_avg_price_per_m2'
    ]
    categorical_features = ['type', 'subtype', 'province']
    boolean_features = [
        'hasLift', 'hasHeatPump', 'hasPhotovoltaicPanels', 'hasAirConditioning',
        'hasVisiophone', 'hasOffice', 'hasSwimmingPool', 'hasFireplace', 'hasAttic',
        'parkingCountIndoor', 'parkingCountOutdoor'
    ]

    feature_cols = numerical_features + categorical_features + boolean_features
    X = df[feature_cols]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = get_model_pipeline(model_name)

    if model_name == 'random_forest':
        param_dist = {
            'model__n_estimators': randint(50, 200),
            'model__max_depth': randint(3, 20),
            'model__min_samples_split': randint(2, 10),
            'model__min_samples_leaf': randint(1, 10),
            'model__max_features': ['auto', 'sqrt', 'log2']
        }
    elif model_name == 'xgboost':
        param_dist = {
            'model__n_estimators': randint(50, 200),
            'model__max_depth': randint(3, 20),
            'model__learning_rate': uniform(0.01, 0.3),
            'model__subsample': uniform(0.6, 0.4),
            'model__colsample_bytree': uniform(0.6, 0.4),
        }
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    search = RandomizedSearchCV(pipeline, param_distributions=param_dist,
                                n_iter=n_iter, cv=5, scoring='neg_mean_absolute_error',
                                verbose=1, n_jobs=-1, random_state=42)

    search.fit(X_train, y_train)

    print("Best params:", search.best_params_)

    y_pred = search.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2:", r2_score(y_test, y_pred))

    return search.best_estimator_, X_test, y_test, y_pred