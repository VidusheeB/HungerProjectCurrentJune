import pandas as pd
import os

def add_metro_area_to_popdata():
    """Add metro_area column to popData.csv by merging with county_to_metro.csv"""
    
    # Read the files
    pop_data_path = os.path.join('src', 'data', 'popData.csv')
    county_metro_path = os.path.join('src', 'data', 'county_to_metro.csv')
    
    if not os.path.exists(pop_data_path):
        print(f"Error: {pop_data_path} not found")
        return False
        
    if not os.path.exists(county_metro_path):
        print(f"Error: {county_metro_path} not found")
        return False
    
    # Read the data
    pop_df = pd.read_csv(pop_data_path)
    county_metro_df = pd.read_csv(county_metro_path)
    
    print(f"Original popData shape: {pop_df.shape}")
    print(f"County-metro mapping shape: {county_metro_df.shape}")
    
    # Check if metro_area column already exists
    if 'metro_area' in pop_df.columns:
        print("metro_area column already exists in popData.csv")
        print("Checking if it needs to be updated...")
        
        # Clean county names for matching
        pop_df['County_Clean'] = pop_df['County'].str.replace(' County', '', regex=False)
        
        # Merge to get the new metro areas
        merged_df = pop_df.merge(
            county_metro_df, 
            left_on='County_Clean', 
            right_on='county', 
            how='left',
            suffixes=('', '_new')
        )
        
        # Check if there are any differences
        if 'metro_area_new' in merged_df.columns:
            differences = merged_df[merged_df['metro_area'] != merged_df['metro_area_new']]
            if not differences.empty:
                print(f"Found {len(differences)} counties with different metro areas:")
                for _, row in differences.iterrows():
                    print(f"  - {row['County']}: {row['metro_area']} -> {row['metro_area_new']}")
                # Update the metro areas
                merged_df['metro_area'] = merged_df['metro_area_new']
                merged_df = merged_df.drop(columns=['metro_area_new'])
            else:
                print("No differences found. Metro areas are up to date.")
                return True
        else:
            print("No new metro areas to merge.")
            return True
    else:
        # Clean county names for matching
        pop_df['County_Clean'] = pop_df['County'].str.replace(' County', '', regex=False)
        
        # Merge the dataframes
        merged_df = pop_df.merge(
            county_metro_df, 
            left_on='County_Clean', 
            right_on='county', 
            how='left'
        )
    
    # Check for any missing metro areas
    missing_metro = merged_df[merged_df['metro_area'].isna()]
    if not missing_metro.empty:
        print(f"Warning: {len(missing_metro)} counties missing metro area mapping:")
        for _, row in missing_metro.iterrows():
            print(f"  - {row['County']}")
    
    # Reorder columns and clean up
    final_df = merged_df[['County', 'Population', 'Population Density', 'metro_area']].copy()
    
    # Save the updated file
    final_df.to_csv(pop_data_path, index=False)
    
    print(f"Updated popData shape: {final_df.shape}")
    print(f"Successfully updated metro_area column in {pop_data_path}")
    
    # Show sample of the updated data
    print("\nSample of updated popData:")
    print(final_df.head(10))
    
    return True

if __name__ == "__main__":
    add_metro_area_to_popdata() 