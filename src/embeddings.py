import requests
import zipfile
import os
from urllib.parse import urlparse

import sys
sys.path.append('..')

relative_target_folder = r".\data\external"


def get_GloVe_embeds():
    zip_url = "https://nlp.stanford.edu/data/glove.6B.zip"
    download_and_extract_zip(zip_url, relative_target_folder)

def get_FastText_embeds():
    zip_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
    download_and_extract_zip(zip_url, relative_target_folder)


def download_and_extract_zip(zip_url, relative_target_folder):
    # Parse the filename from the URL
    zip_filename = os.path.join(relative_target_folder, os.path.basename(urlparse(zip_url).path))

    # Download the zip file
    response = requests.get(zip_url)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Save the zip file with the original name
        with open(zip_filename, "wb") as zip_file:
            zip_file.write(response.content)

        # Extract the contents of the zip file
        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(relative_target_folder)

        print(f"Files extracted to {relative_target_folder}")
        
    else:
        print(f"Failed to download the zip file. Status code: {response.status_code}")