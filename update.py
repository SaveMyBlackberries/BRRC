import os
import requests
import hashlib
import threading

# 下载文件
def download_file(url, destination):
    try:
        response = requests.get(url)
        with open(destination, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {destination}")
    except Exception as e:
        print(f"Error downloading {destination}: {e}")

# 获取文件的SHA256哈希值
def get_file_sha(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return hashlib.sha256(response.content).hexdigest()
        else:
            print(f"Failed to get SHA for {url}")
            return None
    except Exception as e:
        print(f"Error fetching SHA for {url}: {e}")
        return None

# 更新文件
def update_file(file_info, base_path=''):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    local_path = os.path.join(script_dir, base_path, file_info['path'], file_info['name'])
    remote_sha = get_file_sha(file_info['download_url'])

    if remote_sha:
        if os.path.exists(local_path):
            with open(local_path, 'rb') as f:
                local_sha = hashlib.sha256(f.read()).hexdigest()
            if local_sha != remote_sha:
                download_file(file_info['download_url'], local_path)
            else:
                print(f'Skipping {local_path}, up to date.')
        else:
            if not os.path.exists(os.path.dirname(local_path)):
                os.makedirs(os.path.dirname(local_path))
            download_file(file_info['download_url'], local_path)

# 处理特定文件夹中的文件
def process_files_in_folder(folder_url, base_path=''):
    response = requests.get(folder_url)
    if response.status_code == 200:
        files_info = response.json()
        threads = []
        for file_info in files_info:
            if file_info['type'] == 'file':
                # 只更新 infer-web.py 和 update.py 文件
                if file_info['name'] in ['infer-web.py', 'update.py']:
                    file_info['path'] = base_path
                    t = threading.Thread(target=update_file, args=(file_info,))
                    t.start()
                    threads.append(t)
            elif file_info['type'] == 'dir':
                # 如果需要更新特定文件夹内的文件，可以在此逻辑中调整
                folder_path = os.path.join(base_path, file_info['name'])
                process_files_in_folder(file_info['url'], folder_path)
        
        for t in threads:
            t.join()

def main():
    try:
        repo_url = 'https://api.github.com/repos/SaveMyBlackberries/BRRC/contents'
        process_files_in_folder(repo_url)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
