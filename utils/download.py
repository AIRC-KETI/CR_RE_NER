import os
import requests
from tqdm import tqdm

def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/uc?id={}&export=download&confirm=t".format(id)
    session = requests.Session()
    response = session.get(URL, stream=True)
    save_response_content(response, destination)
    decompression(destination)
    

def get_confirm_token(response):
    print(response.cookies.items())
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None
	
def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    pbar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="download")
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                pbar.update(len(chunk))
    pbar.close()
    if total_size_in_bytes != 0 and pbar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

def decompression(destination):
    os.system("tar -xvf {} -C ./".format(destination))
    os.system("rm {}".format(destination))

if __name__ == "__main__":
    OUTPUT_GOOGLE_DRIVE_ID = "1okVoGlrmvqO3Ii12Q3wXVTbOE9w36H7E"
    download_file_from_google_drive(OUTPUT_GOOGLE_DRIVE_ID, "./output.tar")