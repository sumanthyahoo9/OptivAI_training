"""
This test suite looks the validation function written for observations.csv
The test fixtures include a variety of synthetic datasets representing:
Valid observation data with all required columns
Data with missing columns
Data with missing values (NaN)
Data with temporal inconsistencies (invalid months, days, hours)
Data with physical inconsistencies (extreme temperatures, invalid humidity, negative wind speed)
Data with setpoint inconsistencies (heating setpoint >= cooling setpoint)
Data with unrealistic electricity demand (negative or extremely high values)
"""

import unittest
import os
import io
from unittest.mock import patch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the function to test
# Adjust the import path as needed for your project structure
from src.read_parse_data.read_observations import validate_observations


class TestReadObservations(unittest.TestCase):
    """Test cases for validate_observations function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample valid observation data
        self.valid_data = pd.DataFrame(
            {
                "month": [1, 2, 3, 4, 5],
                "day_of_month": [15, 16, 17, 18, 19],
                "hour": [8, 9, 10, 11, 12],
                "outdoor_temperature": [10.5, 12.3, 15.7, 18.2, 20.1],
                "outdoor_humidity": [45.0, 48.5, 50.2, 52.7, 55.0],
                "wind_speed": [5.2, 4.8, 6.1, 7.3, 5.9],
                "wind_direction": [120, 135, 150, 165, 180],
                "diffuse_solar_radiation": [100, 120, 140, 160, 180],
                "direct_solar_radiation": [250, 300, 350, 400, 450],
                "htg_setpoint": [18.0, 18.0, 18.0, 18.0, 18.0],
                "clg_setpoint": [24.0, 24.0, 24.0, 24.0, 24.0],
                "air_temperature": [21.5, 22.0, 22.5, 23.0, 23.5],
                "air_humidity": [35.0, 36.0, 37.0, 38.0, 39.0],
                "people_occupant": [10, 15, 20, 25, 30],
                "co2_emission": [500, 550, 600, 650, 700],
                "HVAC_electricity_demand_rate": [5000, 5500, 6000, 6500, 7000],
                "total_electricity_HVAC": [15000, 16000, 17000, 18000, 19000],
            }
        )

        # Data with missing columns
        self.missing_columns_data = pd.DataFrame(
            {
                "month": [1, 2, 3, 4, 5],
                "day_of_month": [15, 16, 17, 18, 19],
                "hour": [8, 9, 10, 11, 12],
                # Missing several required columns
                "outdoor_temperature": [10.5, 12.3, 15.7, 18.2, 20.1],
                "air_temperature": [21.5, 22.0, 22.5, 23.0, 23.5],
            }
        )

        # Data with missing values
        self.missing_values_data = self.valid_data.copy()
        self.missing_values_data.loc[1, "outdoor_temperature"] = np.nan
        self.missing_values_data.loc[3, "air_humidity"] = np.nan

        # Data with temporal inconsistencies
        self.temporal_inconsistent_data = self.valid_data.copy()
        self.temporal_inconsistent_data.loc[0, "month"] = 0  # Invalid month
        self.temporal_inconsistent_data.loc[1, "day_of_month"] = 32  # Invalid day
        self.temporal_inconsistent_data.loc[2, "hour"] = 24  # Invalid hour

        # Data with physical inconsistencies
        self.physical_inconsistent_data = self.valid_data.copy()
        self.physical_inconsistent_data.loc[0, "outdoor_temperature"] = -60  # Extreme temperature
        self.physical_inconsistent_data.loc[1, "outdoor_humidity"] = 110  # Invalid humidity
        self.physical_inconsistent_data.loc[2, "wind_speed"] = -5  # Negative wind speed

        # Data with setpoint inconsistencies
        self.setpoint_inconsistent_data = self.valid_data.copy()
        self.setpoint_inconsistent_data.loc[0, "htg_setpoint"] = 25.0  # Higher than cooling setpoint

        # Data with unrealistic electricity demand
        self.electricity_inconsistent_data = self.valid_data.copy()
        self.electricity_inconsistent_data.loc[0, "HVAC_electricity_demand_rate"] = -1000  # Negative demand
        self.electricity_inconsistent_data.loc[1, "HVAC_electricity_demand_rate"] = 2e6  # Extremely high demand

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.savefig")
    def test_valid_data(self, mock_savefig, mock_figure, mock_read_csv, mock_stdout):
        """Test validation with valid observation data."""
        # Setup mock
        mock_read_csv.return_value = self.valid_data

        # Call function
        result = validate_observations("valid_file.csv")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Loading valid_file.csv", output)
        self.assertIn("All expected columns are present", output)
        self.assertIn("No missing values detected", output)
        self.assertIn("Month values are in correct range", output)
        self.assertIn("Day values are in correct range", output)
        self.assertIn("Hour values are in correct range", output)
        self.assertIn("is within reasonable range", output)
        self.assertIn("All setpoints satisfy the constraint", output)

        # Verify visualization was attempted
        mock_figure.assert_called()
        mock_savefig.assert_called()

        # Check return value
        pd.testing.assert_frame_equal(result, self.valid_data)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.savefig")
    def test_missing_columns(self, mock_savefig, mock_figure, mock_read_csv, mock_stdout):
        """Test validation with missing columns."""
        # Setup mock
        mock_read_csv.return_value = self.missing_columns_data

        # Call function
        result = validate_observations("missing_columns.csv")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Loading missing_columns.csv", output)
        self.assertIn("Missing columns:", output)

        # Check if it correctly identified some of the missing columns
        self.assertIn("outdoor_humidity", output)
        self.assertIn("wind_direction", output)
        self.assertIn("htg_setpoint", output)

        # Check return value
        pd.testing.assert_frame_equal(result, self.missing_columns_data)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.savefig")
    def test_missing_values(self, mock_savefig, mock_figure, mock_read_csv, mock_stdout):
        """Test validation with missing values."""
        # Setup mock
        mock_read_csv.return_value = self.missing_values_data

        # Call function
        result = validate_observations("missing_values.csv")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Loading missing_values.csv", output)
        self.assertIn("Missing values detected:", output)

        # Check return value
        pd.testing.assert_frame_equal(result, self.missing_values_data)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.savefig")
    def test_temporal_inconsistencies(self, mock_savefig, mock_figure, mock_read_csv, mock_stdout):
        """Test validation with temporal inconsistencies."""
        # Setup mock
        mock_read_csv.return_value = self.temporal_inconsistent_data

        # Call function
        result = validate_observations("temporal_inconsistent.csv")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Loading temporal_inconsistent.csv", output)
        self.assertIn("Warning: Month values out of range", output)
        self.assertIn("Warning: Day values out of range", output)
        self.assertIn("Warning: Hour values out of range", output)

        # Check return value
        pd.testing.assert_frame_equal(result, self.temporal_inconsistent_data)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.savefig")
    def test_physical_inconsistencies(self, mock_savefig, mock_figure, mock_read_csv, mock_stdout):
        """Test validation with physical inconsistencies."""
        # Setup mock
        mock_read_csv.return_value = self.physical_inconsistent_data

        # Call function
        result = validate_observations("physical_inconsistent.csv")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Loading physical_inconsistent.csv", output)
        self.assertIn("Warning: outdoor_temperature has extreme values", output)
        self.assertIn("Warning: outdoor_humidity has out-of-range values", output)
        self.assertIn("Warning: Negative wind speeds detected", output)

        # Check return value
        pd.testing.assert_frame_equal(result, self.physical_inconsistent_data)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.savefig")
    def test_setpoint_inconsistencies(self, mock_savefig, mock_figure, mock_read_csv, mock_stdout):
        """Test validation with setpoint inconsistencies."""
        # Setup mock
        mock_read_csv.return_value = self.setpoint_inconsistent_data

        # Call function
        result = validate_observations("setpoint_inconsistent.csv")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Loading setpoint_inconsistent.csv", output)
        self.assertIn("Warning: Found 1 rows where heating setpoint >= cooling setpoint", output)

        # Check return value
        pd.testing.assert_frame_equal(result, self.setpoint_inconsistent_data)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.savefig")
    def test_electricity_inconsistencies(self, mock_savefig, mock_figure, mock_read_csv, mock_stdout):
        """Test validation with electricity demand inconsistencies."""
        # Setup mock
        mock_read_csv.return_value = self.electricity_inconsistent_data

        # Call function
        result = validate_observations("electricity_inconsistent.csv")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Loading electricity_inconsistent.csv", output)
        self.assertIn("Warning: Negative electricity demand detected", output)
        self.assertIn("Warning: Extremely high electricity demand detected", output)

        # Check return value
        pd.testing.assert_frame_equal(result, self.electricity_inconsistent_data)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    def test_file_error(self, mock_read_csv, mock_stdout):
        """Test validation with file loading error."""
        # Setup mock to raise an exception
        mock_read_csv.side_effect = Exception("Test file error")

        # Call function
        result = validate_observations("non_existent_file.csv")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Loading non_existent_file.csv", output)
        self.assertIn("Error loading file: Test file error", output)

        # Check return value should be None
        self.assertIsNone(result)

    def test_integration(self):
        """Integration test with a temporary CSV file."""
        # Create a temporary CSV file
        temp_file = "temp_test_observations.csv"
        try:
            self.valid_data.to_csv(temp_file, index=False)

            # Call function with actual file but patch matplotlib to avoid creating files
            with (
                patch("matplotlib.pyplot.figure"),
                patch("matplotlib.pyplot.savefig"),
                patch("sys.stdout", new_callable=io.StringIO) as mock_stdout,
            ):
                result = validate_observations(temp_file)

            # Check result
            self.assertIsNotNone(result)
            pd.testing.assert_frame_equal(result, self.valid_data)

            # Check that it printed expected output
            output = mock_stdout.getvalue()
            self.assertIn(f"Loading {temp_file}", output)
            self.assertIn("All expected columns are present", output)

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)


if __name__ == "__main__":
    unittest.main()
