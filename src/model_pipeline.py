from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def get_preprocessor(numerical_features, categorical_features, boolean_features):
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features),
        ('bool', 'passthrough', boolean_features)
    ])

    return preprocessor


def get_model_pipeline(model_name='random_forest'):
    # Define your feature lists here or pass as args
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

    preprocessor = get_preprocessor(numerical_features, categorical_features, boolean_features)

    if model_name == 'random_forest':
        model = RandomForestRegressor(random_state=42)
    elif model_name == 'xgboost':
        model = XGBRegressor(random_state=42, eval_metric='rmse')
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline