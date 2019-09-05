from typing import List, Dict
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup


sv_wiki_base_url = "https://sv.wikipedia.org"
excellent_articles_sitemap = f"{sv_wiki_base_url}/wiki/Wikipedia:Utm%C3%A4rkta_artiklar"
good_articles_sitemap = f"{sv_wiki_base_url}/wiki/Wikipedia:Bra_artiklar"
recommended_articles_sitemap = f"{sv_wiki_base_url}/wiki/Wikipedia:Rekommenderade_artiklar"
previously_listed_articles_sitemap = f"{sv_wiki_base_url}/wiki/Wikipedia:Tidigare_utvalda_artiklar"


def extract_articles_from_wiki_overview_page(page_url: str) -> List[Dict[str, str]]:
    page = requests.get(page_url)
    soup = BeautifulSoup(page.content, "html.parser")
    rows = soup.body.find("div", {"id": "mw-content-text"}).findAll("table")[1].findAll("dl")
    articles = []
    for row in rows:
        articles.extend([{'url': f"{sv_wiki_base_url}{a.get('href')}?action=render",
                          'title': unquote(a.get('href').split("/")[-1])}
                         for a in row.findAll("a")])
    return articles


def extract_articles_from_wiki_list_page(page_url: str) -> List[Dict[str, str]]:
    page = requests.get(page_url)
    soup = BeautifulSoup(page.content, "html.parser")
    links = soup.body.find("div", {"id": "mw-content-text"}).findAll("table", {"class": "wikitable"})[0].findAll("a")
    return [{'url': f"{sv_wiki_base_url}{l.get('href')}?action=render",
             'title': unquote(l.get('href').split("/")[-1])}
            for l in links]


def fetch_articles_to_parse():
    excellent_articles = extract_articles_from_wiki_overview_page(excellent_articles_sitemap)
    good_articles = extract_articles_from_wiki_overview_page(good_articles_sitemap)
    recommended_articles = extract_articles_from_wiki_overview_page(recommended_articles_sitemap)
    previously_listed_articles = extract_articles_from_wiki_list_page(previously_listed_articles_sitemap)

    return excellent_articles + good_articles + recommended_articles + previously_listed_articles
