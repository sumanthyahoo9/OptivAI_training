"""
Tests for the function that validates the rewards obtained by the agent.
Detects Basic Data Issues:
Missing required columns
Missing or null values in the data
Basic data dimensions and structure

Identifies Statistical Anomalies:
Outliers using the IQR (Interquartile Range) method
Extremely large or small values outside expected ranges
Non-finite values (NaN, Inf, -Inf)

Recognizes Pattern Issues:
Constant reward values (no variation)
Suspiciously long sequences of identical rewards

Generates Visualizations:
Creates time series plots and histograms
Saves visualization outputs

The testing approach involves:
Creating Test Data: We built sample datasets representing various scenarios:
Normal, well-formed data
Data with missing columns
Data with outliers
Data with identical or sequential values

Mocking External Dependencies:
We mocked pandas.read_csv to provide controlled test data
We mocked matplotlib functions to prevent actual plot creation
We captured printed output to verify correct reporting

Testing Edge Cases:
File loading errors
Missing required columns
Data with non-finite values
"""

import io
import os
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np

# Import the function to test
# Adjust the import path as needed for your project structure
from src.read_parse_data.read_agent_rewards import validate_rewards


class TestReadAgentRewards(unittest.TestCase):
    """Test cases for validate_rewards function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data for testing
        self.valid_data = pd.DataFrame({"reward": [-1.0, -0.5, 0.0, 0.5, 1.0]})

        self.missing_columns_data = pd.DataFrame({"incorrect_column": [-1.0, -0.5, 0.0, 0.5, 1.0]})

        self.missing_values_data = pd.DataFrame({"reward": [-1.0, np.nan, 0.0, 0.5, 1.0]})

        self.outlier_data = pd.DataFrame({"reward": [-1.0, -0.5, 0.0, 0.5, 1.0, 15.0, -12.0]})  # Contains outliers

        self.non_finite_data = pd.DataFrame({"reward": [-1.0, np.nan, np.inf, -np.inf, 1.0]})  # Contains non-finite values

        self.constant_data = pd.DataFrame({"reward": [0.5, 0.5, 0.5, 0.5, 0.5]})  # Contains identical values

        self.long_sequence_data = pd.DataFrame(
            {"reward": [0.5] * 150 + [0.7] * 150}  # Contains long sequences of identical values
        )

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.savefig")
    def test_valid_data(self, mock_savefig, mock_figure, mock_read_csv, mock_stdout):
        """Test function with valid data."""
        # Setup mock
        mock_read_csv.return_value = self.valid_data

        # Call function
        result = validate_rewards("fake_path.csv")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("All expected columns are present", output)
        self.assertIn("No missing values detected", output)
        self.assertIn("No reward outliers detected using IQR method", output)
        self.assertIn("No extreme rewards detected", output)
        self.assertIn("No non-finite values detected", output)

        # Verify that visualization was attempted - just check it was called, not exactly once
        self.assertTrue(mock_figure.called)
        mock_savefig.assert_called_once_with("reward_analysis.png")

        # Check return value
        pd.testing.assert_frame_equal(result, self.valid_data)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.savefig")
    def test_missing_columns(self, mock_savefig, mock_figure, mock_read_csv, mock_stdout):
        """Test function with missing columns."""
        # Setup mock
        mock_read_csv.return_value = self.missing_columns_data

        # Expect a KeyError since the function doesn't check for column existence
        with self.assertRaises(KeyError):
            _ = validate_rewards("fake_path.csv")

        # Check output messages - we should still see these before the error
        output = mock_stdout.getvalue()
        self.assertIn("Missing columns: ['reward']", output)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.savefig")
    def test_missing_values(self, mock_savefig, mock_figure, mock_read_csv, mock_stdout):
        """Test function with missing values."""
        # Setup mock
        mock_read_csv.return_value = self.missing_values_data

        # Call function
        result = validate_rewards("fake_path.csv")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Missing values detected", output)

        # Check return value
        pd.testing.assert_frame_equal(result, self.missing_values_data)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.savefig")
    def test_outliers(self, mock_savefig, mock_figure, mock_read_csv, mock_stdout):
        """Test function with outlier values."""
        # Setup mock
        mock_read_csv.return_value = self.outlier_data

        # Call function
        result = validate_rewards("fake_path.csv")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Found", output)
        self.assertIn("potential reward outliers using IQR method", output)
        self.assertIn("extremely large or small rewards", output)

        # Check return value
        pd.testing.assert_frame_equal(result, self.outlier_data)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.savefig")
    def test_non_finite_values(self, mock_savefig, mock_figure, mock_read_csv, mock_stdout):
        """Test function with non-finite values."""
        # Setup mock
        mock_read_csv.return_value = self.non_finite_data

        # Call function
        result = validate_rewards("fake_path.csv")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Found", output)
        self.assertIn("non-finite values (NaN, Inf, -Inf)", output)

        # Check return value
        pd.testing.assert_frame_equal(result, self.non_finite_data)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.savefig")
    def test_constant_rewards(self, mock_savefig, mock_figure, mock_read_csv, mock_stdout):
        """Test function with constant reward values."""
        # Setup mock
        mock_read_csv.return_value = self.constant_data

        # Call function
        result = validate_rewards("fake_path.csv")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Warning: All rewards have the same value", output)

        # Check return value
        pd.testing.assert_frame_equal(result, self.constant_data)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.savefig")
    def test_long_sequences(self, mock_savefig, mock_figure, mock_read_csv, mock_stdout):
        """Test function with long sequences of identical rewards."""
        # Setup mock
        mock_read_csv.return_value = self.long_sequence_data

        # Call function
        result = validate_rewards("fake_path.csv")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Found suspicious long sequences of identical rewards", output)

        # Check return value
        pd.testing.assert_frame_equal(result, self.long_sequence_data)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("pandas.read_csv")
    def test_file_error(self, mock_read_csv, mock_stdout):
        """Test function with file loading error."""
        # Setup mock to raise an exception
        mock_read_csv.side_effect = Exception("Test file error")

        # Call function
        result = validate_rewards("non_existent_file.csv")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Error loading file: Test file error", output)

        # Check return value should be None
        self.assertIsNone(result)

    def test_integration(self):
        """Integration test with a temporary CSV file."""
        # Skip the plot generation for this test
        with patch("matplotlib.pyplot.figure"), patch("matplotlib.pyplot.savefig"):
            # Create a temporary CSV file
            temp_file = "temp_test_rewards.csv"
            try:
                self.valid_data.to_csv(temp_file, index=False)

                # Call function with actual file
                with patch("sys.stdout", new_callable=io.StringIO) as _:
                    result = validate_rewards(temp_file)

                # Check result
                self.assertIsNotNone(result)
                pd.testing.assert_frame_equal(result, self.valid_data)

            finally:
                # Clean up temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)


if __name__ == "__main__":
    unittest.main()
