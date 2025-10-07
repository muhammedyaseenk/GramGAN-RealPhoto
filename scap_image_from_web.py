import requests
from bs4 import BeautifulSoup
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import threading

class BingImageScraper:
    def __init__(self, query, max_images=500, download_folder="downloaded_images"):
        self.query = query
        self.max_images = max_images
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.img_urls = []
        self.download_folder = os.path.join(download_folder, query.replace(" ", "_"))
        os.makedirs(self.download_folder, exist_ok=True)
    
    def fetch_image_urls_recurively(self):
        collected_urls = set()
        count_per_page = 35
        for offset in range(0, self.max_images, count_per_page):
            url = f"https://www.bing.com/images/async?q={self.query}&first={offset}&count={count_per_page}&relp={count_per_page}&scenario=ImageBasicHover"
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                images = soup.find_all("img")
                for img in images:
                    img_url = img.get("src") or img.get("data-src")
                    if img_url and img_url.startswith("http") and img_url not in collected_urls:
                        collected_urls.add(img_url)
                    if len(collected_urls) >= self.max_images:
                        break
                if len(collected_urls) >= self.max_images:
                    break
            except Exception as e:
                print(f"[{self.query}] Error fetching URLs at offset {offset}: {e}")
                break
        self.img_urls = list(collected_urls)
        print(f"[{self.query}] Collected {len(self.img_urls)} image URLs.")
        return self.img_urls

    def download_image(self, index_url):
        index, url = index_url
        try:
            img_data = requests.get(url, timeout=10).content
            file_path = os.path.join(self.download_folder, f"img_{self.query.replace(' ', '_')}_{index}.jpg")
            with open(file_path, 'wb') as f:
                f.write(img_data)
            return f"[{self.query}] Downloaded: {file_path}"
        except Exception as e:
            return f"[{self.query}] Failed: {url} ({e})"

    def download_images_parallel(self, max_workers=10):
        if not self.img_urls:
            print(f"[{self.query}] No image URLs found. Run fetch_image_urls_recurively() first.")
            return
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.download_image, (i, url)) for i, url in enumerate(self.img_urls)]
            for future in as_completed(futures):
                print(future.result())

def average_image_dimensions(filenames):
    total_width, total_height = 0, 0
    count = 0
    for file in filenames:
        try:
            with Image.open(file) as img:
                w, h = img.size
                total_width += w
                total_height += h
                count += 1
        except Exception as e:
            print(f"Could not open {file}: {e}")
            # Optionally delete corrupted files
            try:
                os.remove(file)
                print(f"Deleted corrupted file: {file}")
            except Exception as rm_err:
                print(f"Failed to delete {file}: {rm_err}")
    return (total_width / count, total_height / count) if count else (0, 0)

def remove_small_images(filenames, min_width=150, min_height=100):
    for file in filenames:
        try:
            with Image.open(file) as img:
                w, h = img.size
                if w < min_width or h < min_height:
                    img.close()
                    os.remove(file)
                    print(f"Deleted small image: {file} ({w}x{h})")
        except Exception as e:
            print(f"Could not open {file}: {e}")
            try:
                os.remove(file)
                print(f"Deleted corrupted file: {file}")
            except Exception as rm_err:
                print(f"Failed to delete {file}: {rm_err}")

def process_query(query: str, max_workers=10):
    scraper = BingImageScraper(query, max_images=500)
    print(f"[{query}] Fetching image URLs...")
    scraper.fetch_image_urls_recurively()
    print(f"[{query}] Downloading images with {max_workers} workers...")
    scraper.download_images_parallel(max_workers=max_workers)
    # Clean up images smaller than average dimensions
    dirname = scraper.download_folder
    files = os.listdir(dirname)
    filenames = [os.path.join(dirname, f) for f in files]
    avg_w, avg_h = average_image_dimensions(filenames)
    print(f"[{query}] Average image dimensions: {avg_w:.1f} x {avg_h:.1f}")
    remove_small_images(filenames, min_width=int(avg_w), min_height=int(avg_h))
    print(f"[{query}] Finished processing.\n")

def parallel_process_queries(queries: list, max_concurrent_queries=3, max_workers_per_query=10):
    with ThreadPoolExecutor(max_workers=max_concurrent_queries) as executor:
        futures = [executor.submit(process_query, q, max_workers_per_query) for q in queries]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing a query: {e}")

if __name__ == "__main__":
    queries = ["horses", "sunsets", "mountains"]  # Add your search queries here
    max_concurrent_queries = 2  # How many queries run in parallel
    max_workers_per_query = 10  # How many threads download images per query
    
    parallel_process_queries(queries, max_concurrent_queries, max_workers_per_query)
