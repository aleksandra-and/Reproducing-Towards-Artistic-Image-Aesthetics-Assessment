import pandas as pd
import os
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.0.0'
}

dirnames = os.listdir("images/")

def download_image(url, name, proxy):
    output_dir = 'images'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, name)
    try:
        response = requests.get(url, proxies=proxy, headers=headers)
        response.raise_for_status()
        with open(output_path, 'wb') as file:
            file.write(response.content)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")

def get_pics(urls, names, proxies):
    with ThreadPoolExecutor() as executor:
        futures = []
        for idx, url in enumerate(urls):
            if names[idx] in dirnames:
                continue
            proxy = proxies[idx % len(proxies)]
            future = executor.submit(download_image, url, names[idx], proxy)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures)):
            pass

if __name__ == '__main__':
    df = pd.read_csv('image_list.csv')
    urls = df['image_link'].values.tolist()
    names = df['image_name'].values.tolist()
    proxies = [
        {'http': 'http://kdsbuzdh:xiqbf8go5ymv@93.120.32.211:9395', 'https': 'http://kdsbuzdh:xiqbf8go5ymv@93.120.32.211:9395'},
        {'http': 'http://kdsbuzdh:xiqbf8go5ymv@93.120.32.134:9318', 'https': 'http://kdsbuzdh:xiqbf8go5ymv@93.120.32.134:9318'},
        {'http': 'http://kdsbuzdh:xiqbf8go5ymv@93.120.32.43:9227', 'https': 'http://kdsbuzdh:xiqbf8go5ymv@93.120.32.43:9227'},
        {'http': 'http://kdsbuzdh:xiqbf8go5ymv@109.207.130.223:8230', 'https': 'http://kdsbuzdh:xiqbf8go5ymv@109.207.130.223:8230'},
        {'http': 'http://kdsbuzdh:xiqbf8go5ymv@109.207.130.192:8199', 'https': 'http://kdsbuzdh:xiqbf8go5ymv@109.207.130.192:8199'},
        {'http': 'http://kdsbuzdh:xiqbf8go5ymv@194.31.162.171:7687', 'https': 'http://kdsbuzdh:xiqbf8go5ymv@194.31.162.171:7687'},
        {'http': 'http://kdsbuzdh:xiqbf8go5ymv@194.31.162.45:7561', 'https': 'http://kdsbuzdh:xiqbf8go5ymv@194.31.162.45:7561'},
        {'http': 'http://kdsbuzdh:xiqbf8go5ymv@91.246.195.224:6993', 'https': 'http://kdsbuzdh:xiqbf8go5ymv@91.246.195.224:6993'},
        {'http': 'http://kdsbuzdh:xiqbf8go5ymv@64.137.65.254:6933', 'https': 'http://kdsbuzdh:xiqbf8go5ymv@64.137.65.254:6933'},
        {'http': 'http://kdsbuzdh:xiqbf8go5ymv@198.105.111.254:6932', 'https': 'http://kdsbuzdh:xiqbf8go5ymv@198.105.111.254:6932'},
        {'http': 'http://kdsbuzdh:xiqbf8go5ymv@103.37.181.232:6888', 'https': 'http://kdsbuzdh:xiqbf8go5ymv@103.37.181.232:6888'},
        {'http': 'http://kdsbuzdh:xiqbf8go5ymv@45.251.61.134:6852', 'https': 'http://kdsbuzdh:xiqbf8go5ymv@45.251.61.134:6852'},
        {'http': 'http://kdsbuzdh:xiqbf8go5ymv@161.123.65.99:6808', 'https': 'http://kdsbuzdh:xiqbf8go5ymv@161.123.65.99:6808'},
        {'http': 'http://kdsbuzdh:xiqbf8go5ymv@161.123.65.93:6802', 'https': 'http://kdsbuzdh:xiqbf8go5ymv@161.123.65.93:6802'},
        {'http': 'http://kdsbuzdh:xiqbf8go5ymv@45.138.117.25:6797', 'https': 'http://kdsbuzdh:xiqbf8go5ymv@45.138.117.25:6797'},
        {'http': 'http://kdsbuzdh:xiqbf8go5ymv@64.137.65.109:6788', 'https': 'http://kdsbuzdh:xiqbf8go5ymv@64.137.65.109:6788'},
        {'http': 'http://kdsbuzdh:xiqbf8go5ymv@161.123.65.70:6779', 'https': 'http://kdsbuzdh:xiqbf8go5ymv@161.123.65.70:6779'},
        {'http': 'http://kdsbuzdh:xiqbf8go5ymv@45.43.71.156:6754', 'https': 'http://kdsbuzdh:xiqbf8go5ymv@45.43.71.156:6754'},
        {'http': 'http://kdsbuzdh:xiqbf8go5ymv@91.223.126.110:6722', 'https': 'http://kdsbuzdh:xiqbf8go5ymv@91.223.126.110:6722'},
        {'http': 'http://kdsbuzdh:xiqbf8go5ymv@161.123.65.7:6716', 'https': 'http://kdsbuzdh:xiqbf8go5ymv@161.123.65.7:6716'},
    ]

    get_pics(urls, names, proxies)
