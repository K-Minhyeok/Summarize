import requests
from bs4 import BeautifulSoup

URL = "https://www.mk.co.kr/news/ranking/newsall"

def get_article_urls(limit: int = 10):
    res = requests.get(URL)

    soup = BeautifulSoup(res.text, "html.parser")

    articles = []

    for li in soup.select("ul.ak_pnews_grid_5 > li")[:limit]:
        a = li.select_one("a")
        title_span = li.select_one("span.text")

        if not a or not title_span:
            continue

        articles.append({
            "title": title_span.get_text(strip=True),
            "url": a["href"]
        })

    return articles


def crawl_content(url:str):
    res = requests.get(url, headers={
        "User-Agent": "Mozilla/5.0"
    })

    soup = BeautifulSoup(res.text, "html.parser")

    content_div = soup.select_one("div.news_cnt_detail_wrap")

    if not content_div:
        return "Fail to find content"

    paragraphs = content_div.select("p[refid]")

    content = "\n".join(
        p.get_text(strip=True)
        for p in paragraphs
        if p.get_text(strip=True)
    )

    if not content:
        content = "Fail to scrap"

    return content

def get_contents(limit: int):
    infos = get_article_urls(limit)

    for info in infos:
        content = crawl_content(info["url"])
        info["content"] = content 

    return infos

if __name__ == "__main__":
    print(get_contents(5))