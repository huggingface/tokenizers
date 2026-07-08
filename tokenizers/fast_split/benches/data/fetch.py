#!/usr/bin/env python3
"""Fetch big real Wikipedia article text per language for the cl100k bench (gitignored output).
English/Japanese use the repo's ../data/big.txt + unigram_wagahaiwa_nekodearu.txt instead."""
import urllib.request, urllib.parse, json, os
ARTS = [("fr","France"),("ru","Россия"),("el","Ελλάδα"),("he","ישראל"),("ar","مصر"),
        ("hi","भारत"),("th","ประเทศไทย"),("zh","数学"),("ko","대한민국")]
here = os.path.dirname(os.path.abspath(__file__))
for lang, title in ARTS:
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action":"query","format":"json","prop":"extracts","explaintext":"1","titles":title})
    try:
        req = urllib.request.Request(url, headers={"User-Agent":"fast_split-bench/0.1"})
        text = next(iter(json.load(urllib.request.urlopen(req, timeout=30))["query"]["pages"].values())).get("extract","")
        open(f"{here}/{lang}.txt","w").write(text)
        print(f"{lang}: {len(text.encode())} bytes")
    except Exception as e:
        print(f"{lang}: FAILED {e}")
