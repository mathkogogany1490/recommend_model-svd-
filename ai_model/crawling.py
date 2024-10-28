import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    url = '''https://search.naver.com/search.naver?where=nexearch&sm=tab_etc&qvt=0&query=박스오피스'''
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    ul = soup.select_one('#main_pack > div.sc_new.cs_common_module.case_list.color_5._au_movie_list_content_wrap > div.cm_content_wrap > div > div > div.mflick > div._panel_popular._tab_content > div.list_image_info.type_pure_top > div > ul:nth-child(1)')
    lis = ul.find_all('li')
    pop_movies = []
    for idx, li in enumerate(lis):
        mn = ''.join(li.text.split()[1:-1])
        pop_movies.append(mn)
        if idx == 4:
            break
    print(pop_movies)