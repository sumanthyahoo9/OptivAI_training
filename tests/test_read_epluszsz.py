"""
What We're Testing
The function performs three main tasks:
Reading a CSV file containing EnergyPlus zone sizing data
Filtering columns to extract those related to "SPACE5-1"
Returning a DataFrame with the Time column and all SPACE5-1 columns
"""

import unittest
import io
from unittest.mock import patch
import pandas as pd

# Import the function to test
# Adjust the import path as needed for your project structure
from src.read_parse_data.read_epluszsz import parse_epluszsz_data


class TestReadEpluszsz(unittest.TestCase):
    """Test cases for parse_epluszsz_data function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data for testing with SPACE5-1 columns
        self.valid_data = pd.DataFrame(
            {
                "Time": ["01:00", "02:00", "03:00"],
                "SPACE5-1:Temperature [C]": [22.0, 22.5, 23.0],
                "SPACE5-1:Humidity [%]": [45.0, 46.0, 47.0],
                "Other Column": [1, 2, 3],
            }
        )

        # Data with no SPACE5-1 columns
        self.no_space5_data = pd.DataFrame(
            {
                "Time": ["01:00", "02:00", "03:00"],
                "SPACE4-1:Temperature [C]": [22.0, 22.5, 23.0],
                "SPACE3-1:Humidity [%]": [45.0, 46.0, 47.0],
                "Other Column": [1, 2, 3],
            }
        )

        # Empty dataframe
        self.empty_data = pd.DataFrame()

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    def test_valid_data(self, mock_read_csv, mock_stdout):
        """Test function with valid data containing SPACE5-1 columns."""
        # Setup mock
        mock_read_csv.return_value = self.valid_data

        # Call function
        result = parse_epluszsz_data("fake_path.csv")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Successfully loaded the Energy Plus ZSZ csv", output)
        self.assertIn("Found 2 columns for SPACE5-1", output)

        # Check return value - should be a dataframe with Time and SPACE5-1 columns only
        expected_result = pd.DataFrame(
            {
                "Time": ["01:00", "02:00", "03:00"],
                "SPACE5-1:Temperature [C]": [22.0, 22.5, 23.0],
                "SPACE5-1:Humidity [%]": [45.0, 46.0, 47.0],
            }
        )
        pd.testing.assert_frame_equal(result, expected_result)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    def test_no_space5_columns(self, mock_read_csv, mock_stdout):
        """Test function with data that doesn't contain any SPACE5-1 columns."""
        # Setup mock
        mock_read_csv.return_value = self.no_space5_data

        # Call function
        result = parse_epluszsz_data("fake_path.csv")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Successfully loaded the Energy Plus ZSZ csv", output)
        self.assertIn("No data for the above zone was found", output)

        # Check return value - should be None since there are no SPACE5-1 columns
        self.assertIsNone(result)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    def test_empty_data(self, mock_read_csv, mock_stdout):
        """Test function with empty data."""
        # Setup mock
        mock_read_csv.return_value = self.empty_data

        # Call function
        result = parse_epluszsz_data("fake_path.csv")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Successfully loaded the Energy Plus ZSZ csv", output)
        self.assertIn("No data for the above zone was found", output)

        # Check return value - should be None since there are no SPACE5-1 columns
        self.assertIsNone(result)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    def test_file_error(self, mock_read_csv, mock_stdout):
        """Test function with file loading error."""
        # Setup mock to raise an exception
        mock_read_csv.side_effect = Exception("Test file error")

        # Call function
        result = parse_epluszsz_data("non_existent_file.csv")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Error loading or parsing epluszsz.csv: Test file error", output)

        # Check return value - should be None for file error
        self.assertIsNone(result)

    def test_integration(self):
        """Integration test with a temporary CSV file."""
        # Create a temporary CSV file
        temp_file = "temp_test_epluszsz.csv"
        try:
            self.valid_data.to_csv(temp_file, index=False)

            # Call function with actual file
            with patch("sys.stdout", new_callable=io.StringIO) as _:
                result = parse_epluszsz_data(temp_file)

            # Check result is not None
            self.assertIsNotNone(result)

            # Verify correct columns are present
            expected_columns = ["Time", "SPACE5-1:Temperature [C]", "SPACE5-1:Humidity [%]"]
            self.assertListEqual(list(result.columns), expected_columns)

            # Verify data is correctly extracted
            self.assertEqual(len(result), 3)  # 3 rows

        finally:
            # Clean up temporary file
            import os

            if os.path.exists(temp_file):
                os.remove(temp_file)


if __name__ == "__main__":
    unittest.main()
