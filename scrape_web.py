#!/usr/bin/env python3

import pandas as pd
import urllib.request as urlib
from bs4 import BeautifulSoup

def preprocess(text):
    text = text.replace(u'<br/>', '')
    text = text.replace('(<a).*(>).*(</a>)', '')
    text = text.replace('(&amp)', '')
    text = text.replace('(&gt)', '')
    text = text.replace('(&lt)', '')
    text = text.replace(u'\xa0', ' ')
    return text

def get_recipe(page_idx):
    try:
        page_url = f'http://www.thefreshloaf.com/node/{page_idx}'
        page = urlib.urlopen(page_url)
        soup = BeautifulSoup(page, 'html.parser')
        # only process pages that are blog posts, aka contain recipes
        if 'node-type-blog' in soup.body['class']:
            print(f'blog at index {page_idx}')
            soup = soup.prettify()
            soup = soup[soup.find('title'):]
            soup = soup[soup.find('content="')+len('content="'):]
            end = soup.find('"')
            return preprocess(soup[:end])
    except Exception:
        print(f'Page: http://www.thefreshloaf.com/node/{page_idx} not found!')

if __name__ == '__main__':
    df = pd.DataFrame()
    start = 55000
    end = 64820
    with open('recipes-jul2020.txt', 'w') as f:
        for idx in range(start, end):
            recipe = get_recipe(idx)
            if isinstance(recipe, str):
                f.write(recipe)
    f.close()
