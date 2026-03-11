import openpyxl
import os

def create_empty_xlsx(file_path):
    wb = openpyxl.Workbook()
    wb.save(file_path)
    print(f"Created {file_path}, size: {os.path.getsize(file_path)} bytes")

create_empty_xlsx('test_openpyxl.xlsx')
try:
    wb = openpyxl.load_workbook('test_openpyxl.xlsx')
    print("Successfully re-loaded with openpyxl")
except Exception as e:
    print(f"Failed to re-load: {e}")
