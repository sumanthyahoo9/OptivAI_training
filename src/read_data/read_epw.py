"""
The EPW files contain both geographical and weather information
for the site's location.
This is to continuously inform the LLM as to what it's working in.
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def validate_epw_file(file_path):
    """
    Validate an Energy Plus Weather file
    Args:
        file_path (.epw): The path to the EnergyPlus Weather file
    """
    print(f"Validating EPW file: {file_path}")
    
    # Check file existence
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return
    
    # Check file extension
    if not file_path.lower().endswith('.epw'):
        print(f"Warning: File {file_path} does not have .epw extension.")
    
    try:
        # Read the EPW file
        # Skip the first 8 lines which contain header information
        data = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            # Extract header info
            header_lines = lines[:8]
            print(f"\nFound {len(header_lines)} header lines")
            
            # Check location data in header
            if len(header_lines) >= 1:
                location_data = header_lines[0].strip().split(',')
                if len(location_data) >= 4:
                    print(f"Location: {location_data[1]}, {location_data[3]}")
                    print(f"WMO Station: {location_data[5] if len(location_data) > 5 else 'Not available'}")
            
            # Process data lines
            data_lines = lines[8:]
            print(f"Found {len(data_lines)} data lines")
            
            # EPW files should typically have 8760 hours (or 8784 for leap years)
            expected_hours = 8760
            if len(data_lines) not in [8760, 8784]:
                print(f"Warning: Unexpected number of data lines. Expected 8760 or 8784, found {len(data_lines)}")
            
            # Parse data lines
            for line in data_lines:
                data.append(line.strip().split(','))
        
        # Define column names based on EPW format
        columns = [
            'Year', 'Month', 'Day', 'Hour', 'Minute', 
            'Data Source and Uncertainty Flags',
            'Dry Bulb Temperature', 'Dew Point Temperature', 
            'Relative Humidity', 'Atmospheric Station Pressure',
            'Extraterrestrial Horizontal Radiation', 'Extraterrestrial Direct Normal Radiation',
            'Horizontal Infrared Radiation Intensity', 'Global Horizontal Radiation',
            'Direct Normal Radiation', 'Diffuse Horizontal Radiation',
            'Global Horizontal Illuminance', 'Direct Normal Illuminance', 
            'Diffuse Horizontal Illuminance', 'Zenith Luminance',
            'Wind Direction', 'Wind Speed', 'Total Sky Cover', 
            'Opaque Sky Cover', 'Visibility',
            'Ceiling Height', 'Present Weather Observation', 
            'Present Weather Codes', 'Precipitable Water',
            'Aerosol Optical Depth', 'Snow Depth', 'Days Since Last Snowfall',
            'Albedo', 'Liquid Precipitation Depth', 'Liquid Precipitation Quantity'
        ]
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        # Convert numeric columns
        numeric_columns = [
            'Year', 'Month', 'Day', 'Hour', 'Minute',
            'Dry Bulb Temperature', 'Dew Point Temperature', 
            'Relative Humidity', 'Atmospheric Station Pressure',
            'Extraterrestrial Horizontal Radiation', 'Extraterrestrial Direct Normal Radiation',
            'Horizontal Infrared Radiation Intensity', 'Global Horizontal Radiation',
            'Direct Normal Radiation', 'Diffuse Horizontal Radiation',
            'Wind Direction', 'Wind Speed', 'Total Sky Cover', 
            'Visibility', 'Precipitable Water', 'Snow Depth', 
            'Days Since Last Snowfall', 'Albedo', 'Liquid Precipitation Depth'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print("\nMissing values detected:")
            print(missing_values[missing_values > 0])
        else:
            print("\nNo missing values detected in data columns")
        
        # Check basic temporal consistency
        time_issues = 0
        for i in range(len(df)):
            month = df.loc[i, 'Month']
            day = df.loc[i, 'Day']
            hour = df.loc[i, 'Hour']
            
            if not (1 <= month <= 12):
                print(f"Invalid month at row {i+1}: {month}")
                time_issues += 1
            
            if not (1 <= day <= 31):
                print(f"Invalid day at row {i+1}: {day}")
                time_issues += 1
            
            if not (1 <= hour <= 24):
                print(f"Invalid hour at row {i+1}: {hour}")
                time_issues += 1
        
        if time_issues == 0:
            print("All time values are within valid ranges")
        else:
            print(f"Found {time_issues} time-related issues")
        
        # Check key weather variables for physical validity
        print("\nChecking physical validity of key weather variables:")
        
        # Dry Bulb Temperature should be within reasonable range (-70°C to 70°C)
        temp_min = df['Dry Bulb Temperature'].min()
        temp_max = df['Dry Bulb Temperature'].max()
        print(f"Dry Bulb Temperature range: {temp_min:.1f}°C to {temp_max:.1f}°C")
        if temp_min < -70 or temp_max > 70:
            print("Warning: Extreme temperature values detected")
        
        # Relative Humidity should be between 0 and 100%
        rh_min = df['Relative Humidity'].min()
        rh_max = df['Relative Humidity'].max()
        print(f"Relative Humidity range: {rh_min:.1f}% to {rh_max:.1f}%")
        if rh_min < 0 or rh_max > 100:
            print("Warning: Relative humidity values outside valid range (0-100%)")
        
        # Wind Speed should be non-negative and within reasonable range
        ws_min = df['Wind Speed'].min()
        ws_max = df['Wind Speed'].max()
        print(f"Wind Speed range: {ws_min:.1f} m/s to {ws_max:.1f} m/s")
        if ws_min < 0:
            print("Warning: Negative wind speed values detected")
        if ws_max > 50:
            print("Warning: Extremely high wind speed values detected")
        
        # Solar radiation should be non-negative and have reasonable maximum
        if 'Direct Normal Radiation' in df.columns:
            rad_min = df['Direct Normal Radiation'].min()
            rad_max = df['Direct Normal Radiation'].max()
            print(f"Direct Normal Radiation range: {rad_min:.1f} Wh/m² to {rad_max:.1f} Wh/m²")
            if rad_min < 0:
                print("Warning: Negative direct normal radiation values detected")
            if rad_max > 1500:
                print("Warning: Extremely high direct normal radiation values detected")
        
        # Visualize key weather variables
        print("\nCreating weather data visualizations...")
        
        # Create time index for plotting
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']].astype(str).agg('-'.join, axis=1), 
                                     format='%Y-%m-%d-%H', errors='coerce')
        
        plt.figure(figsize=(15, 12))
        
        # Plot temperature
        plt.subplot(3, 1, 1)
        plt.plot(df['Date'], df['Dry Bulb Temperature'])
        plt.title('Dry Bulb Temperature')
        plt.ylabel('Temperature (°C)')
        plt.grid(True)
        
        # Plot relative humidity
        plt.subplot(3, 1, 2)
        plt.plot(df['Date'], df['Relative Humidity'])
        plt.title('Relative Humidity')
        plt.ylabel('Humidity (%)')
        plt.grid(True)
        
        # Plot wind speed
        plt.subplot(3, 1, 3)
        plt.plot(df['Date'], df['Wind Speed'])
        plt.title('Wind Speed')
        plt.ylabel('Speed (m/s)')
        plt.xlabel('Date')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('epw_weather_analysis.png')
        print("Weather visualizations saved to 'epw_weather_analysis.png'")
        
        # Generate monthly statistics
        print("\nGenerating monthly statistics...")
        df['Month'] = pd.to_numeric(df['Month'])
        monthly_stats = df.groupby('Month')['Dry Bulb Temperature'].agg(['mean', 'min', 'max'])
        print(monthly_stats)
        
        # Check for data continuity
        print("\nChecking for data continuity...")
        expected_hours = 24 * df.groupby(['Year', 'Month', 'Day']).size().count()
        actual_hours = len(df)
        if expected_hours != actual_hours:
            print(f"Warning: Expected {expected_hours} hours based on date range, found {actual_hours}")
        else:
            print(f"Data continuity check passed: {actual_hours} hours as expected")
        
        print("\nValidation complete.")
        
    except Exception as e:
        print(f"Error validating EPW file: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate EnergyPlus Weather (EPW) file')
    parser.add_argument('file_path', type=str, help='Path to the EPW file')
    args = parser.parse_args()
    validate_epw_file(args.file_path)