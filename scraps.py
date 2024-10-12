import threading
import requests

def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response.content)
        print(f"{filename} downloaded")

# URLs of two files to be downloaded
urls = ["https://data.ny.gov/api/views/d6yy-54nr/rows.csv?accessType=DOWNLOAD", "https://data.wa.gov/api/views/f6w7-q2d2/rows.csv?accessType=DOWNLOAD"]
filenames = ["Lottery_Powerball_Winning_Numbers__Beginning_2010.csv", "Electric_Vehicle_Population_Data.csv"]

# create and start two threads
for i in range(2):
    t = threading.Thread(target=download_file, args=(urls[i], filenames[i]))
    t.start()