import os
import requests
import tarfile

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

print("Downloading WikiCoherence Corpus...")
download_file_from_google_drive("1Il9mZt111kRAkzy8IXirp_7NUZ2dYAs2", "WikiCoherence.tar.gz")
tar = tarfile.open("WikiCoherence.tar.gz", "r:gz")
tar.extractall()
tar.close()
os.remove("WikiCoherence.tar.gz")

print("Downloading GloVe embeddings...")
os.system("wget http://nlp.stanford.edu/data/glove.840B.300d.zip")
os.system("unzip glove.840B.300d.zip")
os.system("rm glove.840B.300d.zip")
os.system("mv glove.840B.300d.txt data")

print("Downloading infersent pre-trained models...")
os.system("curl -Lo data/infersent1.pkl https://dl.fbaipublicfiles.com/senteval/infersent/infersent1.pkl")
