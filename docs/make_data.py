from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import re
import requests
from typing import List
from urllib.parse import urljoin, urlparse

today = datetime.now()


def get_all_website_links(base_url: str, from_file=True) -> List[str]:
    """Gets all URLs from a base URL

    Alternatively, lots of links have already been downloaded so you can just load from file
    """

    # if we have already scraped the base_url, load its list of URLs locally
    if from_file:
        if base_url == 'https://www.arthur.ai/':
            filename = 'arthur_site_urls.txt'
        elif base_url == 'https://docs.arthur.ai/':
            filename = 'scope_docs_urls.txt'
        elif base_url == 'https://bench.readthedocs.io/en/latest/':
            filename = 'bench_docs_urls.txt'
        elif base_url == 'https://www.arthur.ai/blog':
            filename = 'arthur_blog_urls.txt'
        with open(f'urls/{filename}', 'r') as f:
            return f.readlines()

    # if we have not already scraped the base_url, scrape now
    urls = set()
    domain_name = urlparse(base_url).netloc
    soup = BeautifulSoup(requests.get(base_url).content, "html.parser")
    for a_tag in soup.findAll("a"):
        href = a_tag.attrs.get("href")
        if href == "" or href is None:
            continue
        href = urljoin(base_url, href)
        parsed_href = urlparse(href)
        href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
        parsed_href = urlparse(href)
        if not (bool(parsed_href.netloc) and bool(parsed_href.scheme)):
            continue
        if href in urls:
            continue
        if domain_name not in href:
            continue
        urls.add(href)
    return urls


def get_text_content(url: str) -> str:
    """Gets the text from the page at the provided URL"""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        text = text.replace('|','')
        return text
    except Exception as e:
        print('error; skipping', str(e))
        return ''


def add_examples(df, base_url, content_type):
    """Fetches all links at the base URL, scrapes text from each link and adds text to a dataframe"""
    print(content_type)
    site_urls = get_all_website_links(base_url)
    for url in site_urls:
        url = url.replace('\n', '')
        print(url)
        text_content = get_text_content(url)
        df.loc[len(df)] = [text_content, content_type, url]
        df.to_csv(f"arthur_index_{today.month}{today.day}.csv")


def make_dataframe() -> None:
    """Scrapes data from all URLs and adds to a dataframe"""
    df = pd.DataFrame({}, columns=['text', 'content_type', 'source'])
    add_examples(df, 'https://www.arthur.ai/', 'arthur_site')
    add_examples(df, 'https://docs.arthur.ai/', 'arthur_scope_docs')
    add_examples(df, 'https://bench.readthedocs.io/en/latest/', 'arthur_bench_docs')
    add_examples(df, 'https://www.arthur.ai/blog', 'arthur_blog')


make_dataframe()
