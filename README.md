KGLM-Data
===

This repository contains the scripts used to create the [Linked WikiText-2](https://rloganiv.github.io/linked-wikitext-2) dataset.
These resources can generally be used to add Wikidata entity and relation annotations to English Wikipedia articles.
For an example of what these annotations look like, please check out our [Linked WikiText-2 Demo](https://rloganiv.github.io/linked-wikitext-2/#/explore).

**Warning**:
This is hastily written research code, tailored to building and analyzing the Linked WikiText-2 dataset.
It will likely require modification to be adapted to other purposes.


Installation
===

Python Setup
---

kglm-data requires Python 3.6 (later versions of Python may also work).
The package and its dependencies can be installed by running the following commands:
```
# Install dependencies.
pip install -r requirements.txt

# The following SpaCy model is needed for POS tagging and a couple other things during annotation.
python -m spacy download core_web_en_sm

# Lastly, install the `kglm_data` package.
pip install -e .
```

External Dependencies
---

kglm-data also relies on the following external dependencies:

- [Stanford CoreNLP Server](https://stanfordnlp.github.io/CoreNLP/corenlp-server.html)
- [@nitishgupta's Neural Entity Linker](https://github.com/nitishgupta/neural-el)

Please refer to their documentation for installation instructions.

Data
---

To link to the Wikidata knowledge graph, you will need a copy of the [Wikidata JSON dump](https://www.wikidata.org/wiki/Wikidata:Database_download).
If you plan on using this code to annotate Wikipedia articles, you will also need a list of their titles.


Annotation Pipeline
===

1. Create the `wiki.db` mapping from Wikipedia article titles to Wikidata identifiers:
    ```{bash}
    python kglm_data/build_wiki_db.py [PATH TO WIKIDATA DUMP] --db wiki.db
    ```
    Note: The Wikidata dump does not include redirects. To add them, you will need to download an XML dump of Wikipedia and then run:
    ```{bash}
    python kglm_data/add_redirects.py [PATH TO WIKIPEDIA XML DUMP] wiki.db
    ```
2. Create `alias.db` and `relation.db` which contain the aliases of and relations between entities.
    ```{bash}
    python kglm_data/build_alias_db.py [PATH TO WIKIDATA DUMP] --db alias.db
    python kglm_data/build_relation_db.py [PATH TO WIKIDATA DUMP] --db relation.db
    ```
2. Download the HTML for the Wikipedia articles you would like to annotate:
    ```{bash}
    python kglm_data/parse_wikitext.py [LIST OF ARTICLE TITLES] > raw_articles.jsonl
    ```
3. Extract the text and entity annotations from the article HTML:
    ```{bash}
    python kglm_data/process_html.py raw_articles.jsonl wiki.db > articles.jsonl
    ```
4. Run @nitishgupta's entity linker. WARNING: The following command should be run inside the `neural-el` directory, not the current working directory:
    ```{bash}
    python neuralel_jsonl.py \
        --config=configs/config.ini \
        --model-path=[PATH TO MODEL IN RESOURCES] \
        --moda=ta \
        --input_jsonl=[PATH TO articles.jsonl FROM PREVIOUS STEP]
        --output_jsonl=articles.el.jsonl
        --doc_key='tokens'
        --pretokenized=True
    ```
5. Add coreference. This step requires that you are running a Stanford CoreNLP server on port 9001.
    ```{bash}
    python realm_coref.py articles.el.jsonl > articles.el.coref.jsonl
    ```
6. Leverage the information added from steps 3-5 to produce the final annotated dataset `annotations.jsonl`
    ```{bash}
    python annotate.py articles.el.coref.jsonl annotations.jsonl \
        --alias_db alias.db \
        --relation_db relation.db \
        --wiki_db wiki.db
    ```
Each annotation will be a span which also includes information about where the link comes from. The possible options are:
- WIKI: Link comes from Wikipedia article HTML.
- NEL: Link comes from named entity linker.
- COREF: Link inherited from an entity in coreference cluster.
- KG: Link comes from an exact string match to an alias in the local knowledge graph (please see [our paper](https://arxiv.org/abs/1906.07241) for details).


Tips for Improving Annotation Speed on Small Datasets
---

The initial scope of this project was to annotate a much larger portion of Wikipedia with links to all of Wikidata.
Because this requires a large amount of memory, by default the alias, relation, and wiki databases are stored on-disk in `SqliteDicts`.
At a later point, the scope of the project was reduced so that everything could be processed in memory (e.g., using regular `dicts` and serializing to pickle files) - which is must faster.
This behavior can be enabled by adding the `--in_memory` flag to the above commands.

You can also significantly reduce the size of the alias and relations databases by limiting the entities and properties (a.k.a relations) to those that appear in a predefined list by using the `--entities` and `--properties` arguments.


Tips for Annotating Text Outside of Wikipedia
---

The only steps in the pipeline that explicitly assume that the input is a Wikipedia article are steps 3 and 4.
In principle you should be able to run the remaining scripts on your own data, so long as it is provided in a JSON-Lines file similar to the output of step 4.
Each object should look like:
```{json}
{
    "title": "title",
    "tokens": [
        ["sentence", "1", "tokens"],
        ["sencente", "2", "tokens"],
        ...
    ],
    "entities": [["wikidata_id", start_idx, end_idx], ...]
}
```
Note: the title and entity annotations can be blank.

Tips for Using a Different Entity Linker / Coreference Resolution System
---

You can use whatever entity linking and coreference resolution system you would like, just make sure that they add the expected fields to the above object:

### Entity Linker

Needs to add a `nel` field, which is a list of objects with the following keys: `start`, `end`, `label`, and `score`.

```{json}
    "nel": [
        {"start": start_idx, "end": end_idx, "label": "wikipedia_label", "score": score},
        ...
    ]
```
Note: The indices are assumed to be exclusive, e.g., the mention is `tokens[start:end]`.

### Coreference Resolution

Needs to add a `clusters` field, which is a list of coreference clusters (which are themselves lists of spans).
```{json}
    "clusters": [
        [[start, end], ..., [start, end]],  # 1st coreference cluster
        [[start, end], ..., [start, end]],  # 2nd coreference cluster
        ...
    ]
```
Note: The indices here are assumed to be inclusice, e.g., the mention is `tokens[start:end + 1]`.
