import unittest
import os
import io
import sys
import pandas as pd
import numpy as np
import tempfile
from unittest.mock import patch, mock_open

# Import the function to test
# Adjust the import path as needed for your project structure
from src.read_parse_data.read_epw import validate_epw_file


class TestReadEpw(unittest.TestCase):
    """Test cases for validate_epw_file function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample EPW header and data lines
        self.header_lines = [
            "LOCATION,Denver Centennial Golden Nr,CO,USA,TMY3,724666,39.74,-105.18,-7.0,1829.0\n",
            "DESIGN CONDITIONS,1,Climate Design Data 2009 ASHRAE Handbook,,Heating,1,0.4%,\n",
            "TYPICAL/EXTREME PERIODS,6,Summer - Week Nearest Max Temperature For Period,Extreme,7/16,7/22,Summer - Week Nearest Average Temperature For Period,Typical,6/29,7/5\n",
            "GROUND TEMPERATURES,3,.5,.71,8.15,25.08,27.05,25.83,22.55,18.50,14.47,10.96,8.60,7.48\n",
            "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0\n",
            "COMMENTS 1,TMY3-24666 -- WMO# 724666 -- Source: NSRDB-SUNY -- Lat: 39.742 -- Long: -105.178 -- Time Zone: -7.0\n",
            "COMMENTS 2, -- Elevation: 1829m -- Time Step: 60min -- Processed by NREL (National Renewable Energy Laboratory) -- Publication Date: 2019\n",
            "DATA PERIODS,1,1,Data,Sunday,1/1,12/31\n",
        ]

        # Create sample data lines (valid values)
        self.data_lines = [
            "1991,1,1,1,0,A7A7E8E8*0?9?9?9?9E8*0*0?9?9?9?9*0*0E8*0*0*0,0.0,0.0,60,97299,0,0,0,0,0,0,0,0,0,0,155,2.1,6,5,8046,9,9,999999999,0.0,0.0,0,188,0.000,0.0,0\n",
            "1991,1,1,2,0,A7A7E8E8*0?9?9?9?9E8*0*0?9?9?9?9*0*0E8*0*0*0,-0.6,-0.6,60,97257,0,0,0,0,0,0,0,0,0,0,147,2.1,5,4,8046,9,9,999999999,0.0,0.0,0,188,0.000,0.0,0\n",
            "1991,1,1,3,0,A7A7E8E8*0?9?9?9?9E8*0*0?9?9?9?9*0*0E8*0*0*0,-1.1,-1.1,60,97215,0,0,0,0,0,0,0,0,0,0,138,1.5,4,3,8046,9,9,999999999,0.0,0.0,0,188,0.000,0.0,0\n",
            "1991,1,1,4,0,A7A7E8E8*0?9?9?9?9E8*0*0?9?9?9?9*0*0E8*0*0*0,-2.2,-2.2,60,97173,0,0,0,0,0,0,0,0,0,0,130,2.1,3,2,8046,9,9,999999999,0.0,0.0,0,188,0.000,0.0,0\n",
            "1991,1,1,5,0,A7A7E8E8*0?9?9?9?9E8*0*0?9?9?9?9*0*0E8*0*0*0,-2.2,-2.2,60,97131,0,0,0,0,0,0,0,0,0,0,130,2.1,3,2,8046,9,9,999999999,0.0,0.0,0,188,0.000,0.0,0\n",
        ]

        # Create data lines with extreme values
        self.extreme_data_lines = [
            "1991,1,1,1,0,A7A7E8E8*0?9?9?9?9E8*0*0?9?9?9?9*0*0E8*0*0*0,-80.0,-80.0,-10,97299,0,0,0,0,0,0,0,0,0,0,155,-5.0,6,5,8046,9,9,999999999,-10.0,0.0,0,188,0.000,0.0,0\n",
            "1991,1,1,2,0,A7A7E8E8*0?9?9?9?9E8*0*0?9?9?9?9*0*0E8*0*0*0,80.0,80.0,110,97257,0,0,0,0,0,0,0,0,0,0,147,60.0,5,4,8046,9,9,999999999,2000.0,0.0,0,188,0.000,0.0,0\n",
            "1991,1,1,3,0,A7A7E8E8*0?9?9?9?9E8*0*0?9?9?9?9*0*0E8*0*0*0,0.0,0.0,60,97215,0,0,0,0,0,0,0,0,0,0,138,0.0,4,3,8046,9,9,999999999,0.0,0.0,0,188,0.000,0.0,0\n",
            "1991,1,1,4,0,A7A7E8E8*0?9?9?9?9E8*0*0?9?9?9?9*0*0E8*0*0*0,0.0,0.0,60,97173,0,0,0,0,0,0,0,0,0,0,130,0.0,3,2,8046,9,9,999999999,0.0,0.0,0,188,0.000,0.0,0\n",
            "1991,1,1,5,0,A7A7E8E8*0?9?9?9?9E8*0*0?9?9?9?9*0*0E8*0*0*0,0.0,0.0,60,97131,0,0,0,0,0,0,0,0,0,0,130,0.0,3,2,8046,9,9,999999999,0.0,0.0,0,188,0.000,0.0,0\n",
        ]

        # Data lines with time inconsistencies
        self.time_inconsistent_data_lines = [
            "1991,13,1,1,0,A7A7E8E8*0?9?9?9?9E8*0*0?9?9?9?9*0*0E8*0*0*0,0.0,0.0,60,97299,0,0,0,0,0,0,0,0,0,0,155,2.1,6,5,8046,9,9,999999999,0.0,0.0,0,188,0.000,0.0,0\n",
            "1991,1,32,2,0,A7A7E8E8*0?9?9?9?9E8*0*0?9?9?9?9*0*0E8*0*0*0,-0.6,-0.6,60,97257,0,0,0,0,0,0,0,0,0,0,147,2.1,5,4,8046,9,9,999999999,0.0,0.0,0,188,0.000,0.0,0\n",
            "1991,1,1,25,0,A7A7E8E8*0?9?9?9?9E8*0*0?9?9?9?9*0*0E8*0*0*0,-1.1,-1.1,60,97215,0,0,0,0,0,0,0,0,0,0,138,1.5,4,3,8046,9,9,999999999,0.0,0.0,0,188,0.000,0.0,0\n",
            "1991,1,1,4,0,A7A7E8E8*0?9?9?9?9E8*0*0?9?9?9?9*0*0E8*0*0*0,-2.2,-2.2,60,97173,0,0,0,0,0,0,0,0,0,0,130,2.1,3,2,8046,9,9,999999999,0.0,0.0,0,188,0.000,0.0,0\n",
            "1991,1,1,5,0,A7A7E8E8*0?9?9?9?9E8*0*0?9?9?9?9*0*0E8*0*0*0,-2.2,-2.2,60,97131,0,0,0,0,0,0,0,0,0,0,130,2.1,3,2,8046,9,9,999999999,0.0,0.0,0,188,0.000,0.0,0\n",
        ]

        # Standard file content
        self.file_content = "".join(self.header_lines + self.data_lines)
        self.extreme_file_content = "".join(self.header_lines + self.extreme_data_lines)
        self.time_inconsistent_file_content = "".join(self.header_lines + self.time_inconsistent_data_lines)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.savefig")
    def test_file_exists_and_is_valid(self, mock_savefig, mock_figure, mock_file, mock_exists, mock_stdout):
        """Test validating an existing EPW file with valid data."""
        # Setup mocks
        mock_exists.return_value = True
        mock_file.return_value.__enter__.return_value.readlines.return_value = self.header_lines + self.data_lines

        # Mock matplotlib to avoid actual plot creation
        with patch("pandas.to_datetime") as mock_to_datetime:
            # Setup mock for to_datetime to return a valid Series for plotting
            mock_to_datetime.return_value = pd.Series(pd.date_range("1991-01-01", periods=5, freq="H"))

            # Call function
            validate_epw_file("valid_file.epw")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Validating EPW file: valid_file.epw", output)
        self.assertIn("Found 8 header lines", output)

        # Note: We can't expect full execution with this mock setup, but we can verify starting steps
        self.assertIn("Warning: Unexpected number of data lines", output)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("os.path.exists")
    def test_file_does_not_exist(self, mock_exists, mock_stdout):
        """Test validating a non-existent EPW file."""
        # Setup mock
        mock_exists.return_value = False

        # Call function
        validate_epw_file("non_existent_file.epw")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Validating EPW file: non_existent_file.epw", output)
        self.assertIn("Error: File non_existent_file.epw does not exist.", output)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("os.path.exists")
    def test_file_wrong_extension(self, mock_exists, mock_stdout):
        """Test validating a file with wrong extension."""
        # Setup mock
        mock_exists.return_value = True

        # Call function with non-EPW file
        with patch("builtins.open", mock_open(read_data="")) as m:
            validate_epw_file("file.txt")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Validating EPW file: file.txt", output)
        self.assertIn("Warning: File file.txt does not have .epw extension.", output)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("os.path.exists")
    def test_file_read_error(self, mock_exists, mock_stdout):
        """Test handling a file read error."""
        # Setup mocks
        mock_exists.return_value = True

        # Setup open to raise an exception
        with patch("builtins.open", side_effect=IOError("Mock file read error")):
            mock_open.side_effect = IOError("Mock file read error")
            # Call function
            validate_epw_file("error_file.epw")

        # Check output messages
        output = mock_stdout.getvalue()
        self.assertIn("Validating EPW file: error_file.epw", output)
        self.assertIn("Error validating EPW file:", output)
        self.assertIn("Mock file read error", output)

    def test_extreme_weather_values(self):
        """Test validating an EPW file with extreme weather values using real file."""
        # Create a temporary file with extreme values
        with tempfile.NamedTemporaryFile(suffix=".epw", mode="w", delete=False) as temp_file:
            temp_file.write(self.extreme_file_content)
            temp_file_path = temp_file.name

        try:
            # Redirect stdout to capture output
            captured_output = io.StringIO()
            original_stdout = sys.stdout
            sys.stdout = captured_output

            # Patch matplotlib functions to avoid actual figure creation
            with patch("matplotlib.pyplot.figure"), patch("matplotlib.pyplot.savefig"):
                # Call function with the test file
                validate_epw_file(temp_file_path)

            # Restore stdout
            sys.stdout = original_stdout

            # Check output for expected messages
            output = captured_output.getvalue()
            self.assertIn("Warning: Extreme temperature values detected", output)
            self.assertIn("Warning: Relative humidity values outside valid range", output)
            self.assertIn("Warning: Negative wind speed values detected", output)

        finally:
            # Clean up
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_time_consistency_issues(self):
        """Test validating an EPW file with time consistency issues using real file."""
        # Create a temporary file with time inconsistencies
        with tempfile.NamedTemporaryFile(suffix=".epw", mode="w", delete=False) as temp_file:
            temp_file.write(self.time_inconsistent_file_content)
            temp_file_path = temp_file.name

        try:
            # Redirect stdout to capture output
            captured_output = io.StringIO()
            original_stdout = sys.stdout
            sys.stdout = captured_output

            # Patch matplotlib functions to avoid actual figure creation
            with patch("matplotlib.pyplot.figure"), patch("matplotlib.pyplot.savefig"):
                # Call function with the test file
                validate_epw_file(temp_file_path)

            # Restore stdout
            sys.stdout = original_stdout

            # Check output for expected messages
            output = captured_output.getvalue()
            self.assertIn("Invalid month at row 1:", output)
            self.assertIn("Invalid day at row 2:", output)
            self.assertIn("Invalid hour at row 3:", output)

        finally:
            # Clean up
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_integration_with_small_test_file(self):
        """Integration test with a small generated EPW file."""
        # Create a temporary EPW file
        with tempfile.NamedTemporaryFile(suffix=".epw", mode="w", delete=False) as temp_file:
            # Write header lines
            for header_line in self.header_lines:
                temp_file.write(header_line)

            # Write a few data lines
            for data_line in self.data_lines:
                temp_file.write(data_line)

            temp_file_path = temp_file.name

        try:
            # Redirect stdout to capture output
            captured_output = io.StringIO()
            original_stdout = sys.stdout
            sys.stdout = captured_output

            # Patch matplotlib functions to avoid actual figure creation
            with patch("matplotlib.pyplot.figure"), patch("matplotlib.pyplot.savefig"):
                # Call function with the test file
                validate_epw_file(temp_file_path)

            # Restore stdout
            sys.stdout = original_stdout

            # Check output for expected messages
            output = captured_output.getvalue()
            self.assertIn(f"Validating EPW file: {temp_file_path}", output)
            self.assertIn("Found 8 header lines", output)
            self.assertIn("Warning: Unexpected number of data lines", output)

        finally:
            # Clean up
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)


if __name__ == "__main__":
    unittest.main()
