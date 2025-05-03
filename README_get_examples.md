# HVAC Training Examples Generator

This guide will help you generate high-quality training examples for HVAC control using your space5_training_data.csv file and OpenAI's API. Follow these simple steps even if you have limited coding experience.

## 1. Get an OpenAI API Key

1. **Create an OpenAI account**:
   - Go to [https://platform.openai.com/signup](https://platform.openai.com/signup)
   - Sign up with your email or Google/Microsoft account

2. **Generate an API key**:
   - After logging in, click on your account icon in the top-right corner
   - Select "View API keys" from the dropdown menu
   - Click on the "+ Create new secret key" button
   - Give your key a name like "HVAC Training Generator"
   - Click "Create secret key"
   - **IMPORTANT**: Copy and save this key immediately in a secure place! You won't be able to see it again.

3. **Add payment method**:
   - Click on "Billing" in the left sidebar
   - Add a payment method if you haven't already
   - Note: GPT-4 will cost approximately $0.03-0.06 per example, so 500 examples ≈ $15-30

## 2. Install Required Software

1. **Install Python** (if not already installed):
   - Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - Download and install Python 3.9 or newer
   - During installation, check the box "Add Python to PATH"

2. **Install a code editor** (if needed):
   - Download and install Visual Studio Code: [https://code.visualstudio.com/download](https://code.visualstudio.com/download)

## 3. Set Up Your Working Environment

1. **Create a project folder**:
   - Create a new folder on your computer called "hvac-example-generator"
   - Move your `space5_training_data.csv` file into this folder

2. **Open a terminal or command prompt**:
   - On Windows: Press Windows+R, type "cmd" and press Enter
   - On Mac: Press Command+Space, type "terminal" and press Enter

3. **Navigate to your project folder**:
   ```
   cd path/to/hvac-example-generator
   ```

4. **Install required libraries**:
   ```
   pip install pandas numpy openai
   ```

## 4. Create the Script File

1. **Open your code editor** (like VS Code)

2. **Create a new file** called `generate_examples.py` in your project folder or use the existing script

3. **Copy and paste the code below** into the file:
    The script you need to run is named "create_queries_fine_tuning.py"

## 5. Run the Script

1. **Save the script**:
   - Save the file in your code editor

2. **Run the script** from your terminal or command prompt:
   ```
   python generate_examples.py --data space5_training_data.csv --num-examples 100 --api-key YOUR_API_KEY_HERE
   ```
   
   - Replace `YOUR_API_KEY_HERE` with your actual OpenAI API key
   - Adjust `--num-examples` to generate more or fewer examples (start small to test)
   - You can use `--model gpt-4` if you want higher quality examples (costs more)
   - You can specify a different output file with `--output my_examples.jsonl`

3. **Monitor the generation process**:
   - The script will show its progress as it generates examples
   - It will tell you when it's done and where it saved the results

## 6. Review the Generated Examples

1. **Check the output file**:
   - Open the generated JSONL file (`llm_enhanced_training_examples.jsonl` by default)
   - You can use any text editor to view it, or tools like VS Code

2. **Verify quality**:
   - Check a few examples to ensure they look reasonable
   - Each entry should have a query, response, and contextual data

## 7. Use the Examples for Fine-Tuning

The generated JSONL file is ready to be used for fine-tuning a language model. You can use it with various open-source LLM fine-tuning frameworks.

## Troubleshooting

- **API Key Error**: Ensure your OpenAI API key is correct and you have billing set up
- **Module not found errors**: Run `pip install pandas numpy openai` again
- **CSV loading error**: Make sure your CSV file is in the correct location and is properly formatted
- **Rate limit errors**: Try generating fewer examples or waiting between attempts

## Cost Considerations

- GPT-3.5 Turbo: Approximately $0.002 per example
- GPT-4: Approximately $0.03-0.06 per example
- Start with a small number of examples to test before generating hundreds

## Tips

- If you need to stop the script, you can press `Ctrl+C` in the terminal
- The generated examples are randomly distributed across different query types
- For the highest quality examples, use `--model gpt-4` (costs more but produces better results)
- You can run the script multiple times to generate different batches of examples