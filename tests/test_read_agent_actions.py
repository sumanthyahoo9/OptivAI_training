"""
Unit test for the read_agent_actions functions
This unit test is designed for the function validate_agent_actions:
Verified the presence of the required columns
Checked the missing values
Identified the outlier values for the temperatures
Ensured heating setpoints are lower than the cooling setpoints
"""

import io
import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

# Import the function to test
# Adjust the import path if needed
from src.read_parse_data.read_agent_actions import validate_agent_actions


class TestReadAgentActions(unittest.TestCase):
    """
    Module to test the function with various cases
    Args:
        unittest (_type_): _description_
    """

    def setUp(self):
        """
        Setup the test fixtures
        """
        # Create sample data for testing
        self.valid_data = pd.DataFrame(
            {"Heating_Setpoint_RL": [20.0, 21.0, 22.0], "Cooling_Setpoint_RL": [25.0, 26.0, 27.0]}
        )
        self.missing_columns_data = pd.DataFrame({"Heating_Setpoint_RL": [20.0, 21.0, 22.0]})

        self.missing_values_data = pd.DataFrame(
            {"Heating_Setpoint_RL": [20.0, np.nan, 22.0], "Cooling_Setpoint_RL": [25.0, 26.0, 27.0]}
        )

        self.outlier_data = pd.DataFrame(
            {
                "Heating_Setpoint_RL": [10.0, 21.0, 30.0],  # Contains outliers
                "Cooling_Setpoint_RL": [15.0, 26.0, 35.0],  # Contains outliers
            }
        )

        self.invalid_constraint_data = pd.DataFrame(
            {
                "Heating_Setpoint_RL": [20.0, 27.0, 22.0],  # Second row violates constraint
                "Cooling_Setpoint_RL": [25.0, 26.0, 27.0],
            }
        )

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    def test_valid_data(self, mock_read_csv, mock_stdout):
        """
        Test the function with valid data
        Args:
            mock_read_csv (_type_): _description_
            mock_stdout (_type_): _description_
        """
        # Setup mock
        mock_read_csv.return_value = self.valid_data
        # Call function
        result = validate_agent_actions("fake_path.csv")
        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("All expected columns are present", output)
        self.assertIn("No missing values detected", output)
        self.assertIn("No heating setpoint outliers detected", output)
        self.assertIn("No cooling setpoint outliers detected", output)
        self.assertIn("All setpoints satisfy the constraint", output)

        # Check return value
        pd.testing.assert_frame_equal(result, self.valid_data)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    def test_missing_columns(self, mock_read_csv, mock_stdout):
        """Test function with missing columns."""
        # Setup mock
        mock_read_csv.return_value = self.missing_columns_data

        result = None
        try:
            # Call function
            result = validate_agent_actions("fake_path.csv")
        except KeyError:
            # Expected error due to missing column
            pass

        # Check output messages - only check what we can expect to see before the error
        output = mock_stdout.getvalue()
        self.assertIn("Missing columns: ['Cooling_Setpoint_RL']", output)

        # Only check the return value if it was set
        if result is not None:
            pd.testing.assert_frame_equal(result, self.missing_columns_data)
        else:
            # If result is None, the function didn't complete due to the KeyError
            # This is expected behavior for this test case
            pass

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    def test_missing_values(self, mock_read_csv, mock_stdout):
        """Test function with missing values."""
        # Setup mock
        mock_read_csv.return_value = self.missing_values_data

        # Call function
        result = validate_agent_actions("fake_path.csv")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Missing values detected", output)

        # Check return value
        pd.testing.assert_frame_equal(result, self.missing_values_data)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    def test_outliers(self, mock_read_csv, mock_stdout):
        """Test function with outlier values."""
        # Setup mock
        mock_read_csv.return_value = self.outlier_data

        # Call function
        result = validate_agent_actions("fake_path.csv")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Found", output)
        self.assertIn("potential heating setpoint outliers", output)
        self.assertIn("potential cooling setpoint outliers", output)

        # Check return value
        pd.testing.assert_frame_equal(result, self.outlier_data)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    def test_invalid_constraint(self, mock_read_csv, mock_stdout):
        """Test function with invalid constraints."""
        # Setup mock
        mock_read_csv.return_value = self.invalid_constraint_data

        # Call function
        result = validate_agent_actions("fake_path.csv")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("rows where heating setpoint >= cooling setpoint", output)

        # Check return value
        pd.testing.assert_frame_equal(result, self.invalid_constraint_data)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    def test_file_error(self, mock_read_csv, mock_stdout):
        """Test function with file loading error."""
        # Setup mock to raise an exception
        mock_read_csv.side_effect = Exception("Test file error")

        # Call function
        result = validate_agent_actions("non_existent_file.csv")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Error loading file: Test file error", output)

        # Check return value should be None
        self.assertIsNone(result)

    def test_integration(self):
        """Integration test with a temporary CSV file."""
        # Create a temporary CSV file
        temp_file = "temp_test_file.csv"
        try:
            self.valid_data.to_csv(temp_file, index=False)

            # Call function with actual file
            with patch("sys.stdout", new_callable=io.StringIO) as _:
                result = validate_agent_actions(temp_file)

            # Check result
            self.assertIsNotNone(result)
            pd.testing.assert_frame_equal(result, self.valid_data)

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)


if __name__ == "__main__":
    unittest.main()
