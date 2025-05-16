import pandas as pd
from feature_engineering import map_epc_score, add_postcode_price_mean, encode_building_condition, encode_flood_zone

def cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # Drop unnecessary columns
    df = df.drop(columns=["Unnamed: 0", "url", "id", "monthlyCost", "accessibleDisabledPeople", "hasBalcony"])

    drop_cols = [
        'roomCount', 'diningRoomSurface', 'streetFacadeWidth', 'kitchenSurface', 'hasBasement', 'hasArmoredDoor',
        'floorCount', 'hasDiningRoom', 'hasDressingRoom', 'gardenSurface', 'terraceSurface', 'livingRoomSurface',
        'gardenOrientation', 'heatingType', 'kitchenType', 'terraceOrientation'
    ]
    df = df.drop(columns=drop_cols)

    # Drop rows with missing target or essential features
    df = df.dropna(axis=0, subset=['price','habitableSurface','bedroomCount','bathroomCount','epcScore'])

    # Convert boolean columns to binary (0/1)
    binary_cols = [
        'hasLift', 'hasHeatPump', 'hasPhotovoltaicPanels', 'hasAirConditioning', 'hasVisiophone', 'hasOffice',
        'hasSwimmingPool', 'hasFireplace', 'hasAttic', 'parkingCountIndoor', 'parkingCountOutdoor'
    ]
    for col in binary_cols:
        df[col] = df[col].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)

    # facadeCount fixes
    df = df[df['facedeCount'] <= 4]  # Remove rows with more than 4 facades
    df['facadeCount'] = df['facedeCount']
    df = df.drop(columns='facedeCount')

    apartment_subtypes = ['APARTMENT', 'FLAT_STUDIO', 'GROUND_FLOOR', 'PENTHOUSE', 'APARTMENT_BLOCK']
    df.loc[df['subtype'].isin(apartment_subtypes), 'facadeCount'] = df.loc[df['subtype'].isin(apartment_subtypes), 'facadeCount'].fillna(1)

    house_subtypes = ['HOUSE', 'VILLA', 'DUPLEX', 'TOWN_HOUSE', 'MANSION']
    df.loc[df['subtype'].isin(house_subtypes), 'facadeCount'] = df.loc[df['subtype'].isin(house_subtypes), 'facadeCount'].fillna(3)

    larger_house_subtypes = ['EXCEPTIONAL_PROPERTY', 'BUNGALOW', 'COUNTRY_COTTAGE', 'TRIPLEX', 'CHALET', 'CASTLE', 'MANOR_HOUSE']
    df.loc[df['subtype'].isin(larger_house_subtypes), 'facadeCount'] = df.loc[df['subtype'].isin(larger_house_subtypes), 'facadeCount'].fillna(4)

    other_subtypes = ['MIXED_USE_BUILDING', 'SERVICE_FLAT', 'KOT', 'FARMHOUSE', 'LOFT', 'OTHER_PROPERTY']
    df.loc[df['subtype'].isin(other_subtypes), 'facadeCount'] = df.loc[df['subtype'].isin(other_subtypes), 'facadeCount'].fillna(2)

    return df


def apply_feature_limits(df: pd.DataFrame, limits: dict) -> pd.DataFrame:
    df_filtered = df.copy()
    for feature, (min_val, max_val) in limits.items():
        df_filtered = df_filtered[(df_filtered[feature] >= min_val) & (df_filtered[feature] <= max_val)]
    return df_filtered


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Feature limits for filtering outliers
    limits = {
        'price': (100000, 800000),
        'habitableSurface': (20, 500),
        'bathroomCount': (1, 10),
        'bedroomCount': (1, 10),
        # Add other limits if needed
    }

    df = map_epc_score(df)
    df = cleaning(df)
    df = add_postcode_price_mean(df)
    df = encode_building_condition(df)
    df = encode_flood_zone(df)
    df = apply_feature_limits(df, limits)
    return df