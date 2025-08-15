import zipfile
import sys

zip_path = sys.argv[1]
extract_to = sys.argv[2]

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)