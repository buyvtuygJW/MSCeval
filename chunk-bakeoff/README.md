# chunk-bakeoff

```
pip install -r requirements.txt && python -m spacy download en_core_web_sm
docker exec <pg-container> psql -U <user> -d <db> -Atc "SELECT COALESCE(json_agg(json_build_object('document', d.filename, 'page', p.page_number, 'text', p.text)), '[]') FROM document_pages p JOIN documents d ON d.id = p.document_id" > pages.json
python run.py --pages pages.json --bench ../benchmarkv2.yaml
```

`chunking/` and `app_embeddings.py` are verbatim app copies (only the logger import differs); `chunking/structure_aware.py` is the new third strategy, drop-in shaped for `rag-service/chunking/`.
