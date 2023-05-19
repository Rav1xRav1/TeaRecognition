from icrawler.builtin import BingImageCrawler

crawler = BingImageCrawler(storage={"root_dir": "images"})
crawler.crawl(keyword="keyword", max_num=100)