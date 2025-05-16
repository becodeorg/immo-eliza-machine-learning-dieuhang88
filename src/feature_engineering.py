# src/feature_engineering.py
import pandas as pd

def map_epc_score(df: pd.DataFrame) -> pd.DataFrame:
    epc_invalid = ['C_A', 'F_C', 'G_C', 'D_C', 'F_D', 'E_C', 'G_E', 'E_D', 'C_B', 'X', 'G_F']
    df = df[~df['epcScore'].isin(epc_invalid)].copy()

    wallonia_provinces = ['Liège', 'Walloon Brabant', 'Namur', 'Hainaut', 'Luxembourg']
    flanders_provinces = ['Antwerp', 'Flemish Brabant', 'East Flanders', 'West Flanders', 'Limburg']

    epc_maps = {
        "Wallonia": {'A++': 0, 'A+': 30, 'A': 65, 'B': 125, 'C': 200, 'D': 300, 'E': 375, 'F': 450, 'G': 510},
        "Flanders": {'A++': 0, 'A+': 0, 'A': 50, 'B': 150, 'C': 250, 'D': 350, 'E': 450, 'F': 500, 'G': 510},
        "Brussels": {'A++': 0, 'A+': 0, 'A': 45, 'B': 75, 'C': 125, 'D': 175, 'E': 250, 'F': 300, 'G': 350}
    }

    def map_score(row):
        if row['province'] in wallonia_provinces:
            return epc_maps['Wallonia'].get(row['epcScore'], None)
        elif row['province'] in flanders_provinces:
            return epc_maps['Flanders'].get(row['epcScore'], None)
        elif row['province'] == 'Brussels':
            return epc_maps['Brussels'].get(row['epcScore'], None)
        return None

    df.loc[:, 'epc_enum'] = df.apply(map_score, axis=1)
    return df

def add_postcode_price_mean(df: pd.DataFrame) -> pd.DataFrame:
    df['price_per_m2'] = df['price'] / df['habitableSurface']

    postcode_price = df.groupby('postCode')['price_per_m2'].mean().reset_index()
    postcode_price.rename(columns={'price_per_m2': 'postcode_avg_price_per_m2'}, inplace=True)

    df = df.merge(postcode_price, on='postCode', how='left')
    return df

def encode_building_condition(df: pd.DataFrame) -> pd.DataFrame:
    condition_mapping = {
        'AS_NEW': 0, 'JUST_RENOVATED': 1, 'GOOD': 2,
        'TO_RENOVATE': 3, 'TO_RESTORE': 4, 'TO_BE_DONE_UP': 5
    }
    df['buildingCondition_mapping'] = df['buildingCondition'].map(condition_mapping)
    return df

def encode_flood_zone(df: pd.DataFrame) -> pd.DataFrame:
    flood_mapping = {
        "NON_FLOOD_ZONE": 0,
        "POSSIBLE_N_CIRCUMSCRIBED_WATERSIDE_ZONE": 1,
        "CIRCUMSCRIBED_WATERSIDE_ZONE": 2,
        "POSSIBLE_N_CIRCUMSCRIBED_FLOOD_ZONE": 3,
        "POSSIBLE_FLOOD_ZONE": 4,
        "CIRCUMSCRIBED_FLOOD_ZONE": 5,
        "RECOGNIZED_FLOOD_ZONE": 6,
        "RECOGNIZED_N_CIRCUMSCRIBED_WATERSIDE_FLOOD_ZONE": 7,
        "RECOGNIZED_N_CIRCUMSCRIBED_FLOOD_ZONE": 8
    }
    df['floodZoneType_mapping'] = df['floodZoneType'].map(flood_mapping)
    return df