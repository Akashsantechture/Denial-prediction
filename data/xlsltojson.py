import pandas as pd

# 1. Define your file paths
excel_file_path = "C:\\Users\\akash.subramani.SANTECHTURE\\Downloads\\Rules List- ML model.xlsx"  # Replace with your actual file name
json_file_path = 'output.json'

# 2. Read the Excel file
# pandas automatically picks up headers: SI.NO, RNum, Name, Package, etc.
df = pd.read_excel(excel_file_path)

# (Optional) Clean up column names if needed, or keep them as is
# df.columns = df.columns.str.strip() 

# 3. Convert the DataFrame to JSON format ('records' creates a clean list of dictionaries)
df.to_json(json_file_path, orient='records', indent=4)

print(f"Successfully converted '{excel_file_path}' to '{json_file_path}'!")