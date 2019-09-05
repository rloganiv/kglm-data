KGLM-Data
===

Resources used to help compile the [Linked WikiText-2](https://rloganiv.github.io/linked-wikitext-2) dataset.

## Dataset construction for swedish:

####Data: 

* Wikidata dump [latest-all.json.gz](https://dumps.wikimedia.org/wikidatawiki/entities/)
* Wikipedia articles ([Featured articles](https://sv.wikipedia.org/wiki/Wikipedia:Utmärkta_artiklar),
                          [Good articles](https://sv.wikipedia.org/wiki/Wikipedia:Bra_artiklar),
                          [Recommended articles](https://sv.wikipedia.org/wiki/Wikipedia:Rekommenderade_artiklar),
                          [Previously selected articles](https://sv.wikipedia.org/wiki/Wikipedia:Tidigare_utvalda_artiklar))
                          
####Scripts
To list urls of articles:
```bash
list_wanted_wiki_articles articles.json
```
To parse wikitext:
```bash
python kglm_data/parse_wikitext.py articles.json data/sv_dump -j 2 --language 'sv'
```
To build Wiki DB:
```bash
python kglm_data/build_wiki_db.py latest-all.json.gz --db data/wiki.pkl --in_memory --language sv
```
You can dump wiki db as a pickle file by using `--in_memory` argument.

To add redirects:
```bash
python kglm_data/add_redirects.py sv_dump data/wiki.pkl --in_memory 
```
To process articles html:
```bash
python kglm_data/process_html.py data/sv_dump --wiki_db data/wiki.pkl --language sv >articles_processed_entities.jsonl
```
To add co-reference:
```bash
add_coref_clusters data/articles_processed_entities.jsonl data/articles_coref.jsonl --spacy_model_dir /spacy_dir -j 0
```
Here `spacy_dir` is an argument which allows for loading spacy models from disk. If path is a string (e.g. “en”), spacy will download the english model.


To built a list of entities:
```bash
python kglm_data/list_entities.py data/articles_nel.jsonl >data/entities.txt
```
To build Relation DB:
```bash
python kglm_data/build_relation_db.py latest-all.json.gz --db data/relation.pkl -e data/entities.txt --reverse --in_memory --language sv
```
The `--reverse` argument .

To build Alias DB:
```bash
python kglm_data/build_alias_db.py latest-all.json.gz --db data/alias.pkl -e data/entitites.txt --in_memory --language sv
```
To annotate:
```bash
python kglm_data/annotate.py /data/articles_nel.jsonl /data/articles_annotated.jsonl -j 1 --alias_db data/alias.pkl --relation_db data/relation.pkl --wiki_db data/wiki.pkl --in_memory -m -c 500 --language sv --spacy_model_path spacy_dir
```
