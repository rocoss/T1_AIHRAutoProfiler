#!/usr/bin/env python3

import time
import xml.etree.ElementTree as ET
import re
import csv
import json
from collections import Counter
from typing import List, Dict, Any
from pathlib import Path
from fastapi import FastAPI, Query
import uvicorn
from opensearchpy import OpenSearch, helpers

OPENSEARCH_URL = "http://localhost:9200"
INDEX_NAME = "products"
XML_FILE = "data/catalog_products.xml"
QUERIES_FILE = "data/prefix_queries.csv"
REPORTS_DIR = Path("reports")

client = None
app = FastAPI(title="Prefix Search API")

QWERTY = "qwertyuiop[]asdfghjkl;'zxcvbnm,./QWERTYUIOP{}ASDFGHJKL:\"ZXCVBNM<>?"
JCUKEN = "йцукенгшщзхъфывапролджэячсмитьбю.ЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ,"
qwerty_to_jcuken = dict(zip(QWERTY, JCUKEN))
jcuken_to_qwerty = dict(zip(JCUKEN, QWERTY))

def connect_opensearch():
    global client
    print("[INIT] Подключение к OpenSearch...")

    for attempt in range(30):
        try:
            client = OpenSearch(hosts=[OPENSEARCH_URL], timeout=30)
            if client.ping():
                info = client.info()
                version = info.get('version', {}).get('number', 'unknown')
                print(f"[INIT] OpenSearch v{version} подключен")
                return True
        except Exception as e:
            if attempt == 0:
                print(f"[INIT] Попытка подключения {attempt+1}/30...")

        if attempt < 29:
            time.sleep(2)

    print("[INIT] OpenSearch не доступен")
    return False

def delete_old_index():
    try:
        if client.indices.exists(index=INDEX_NAME):
            print(f"[INIT] Удаляю старый индекс...")
            client.indices.delete(index=INDEX_NAME)
            time.sleep(1)
    except:
        pass

def create_index():
    print(f"[INIT] Создаю индекс '{INDEX_NAME}'...")

    settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "edge_ngram_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "edge_ngram_filter"]
                    }
                },
                "filter": {
                    "edge_ngram_filter": {
                        "type": "edge_ngram",
                        "min_gram": 1,
                        "max_gram": 15
                    }
                }
            },
            "index": {"max_ngram_diff": 14}
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "name": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {
                        "ngram": {
                            "type": "text",
                            "analyzer": "edge_ngram_analyzer",
                            "search_analyzer": "standard"
                        }
                    }
                },
                "category": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "brand": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "keywords": {"type": "text", "analyzer": "edge_ngram_analyzer", "search_analyzer": "standard"},
                "description": {"type": "text"},
                "weight_value": {"type": "float"},
                "weight_unit": {"type": "keyword"},
                "price": {"type": "float"},
                "package_size": {"type": "integer"}
            }
        }
    }

    try:
        client.indices.create(index=INDEX_NAME, body=settings)
        print(f"[INIT] Индекс создан")
        return True
    except Exception as e:
        print(f"[INIT] Ошибка создания индекса: {e}")
        return False

def load_documents():
    print(f"[INIT] Загружаю товары из {XML_FILE}...")

    try:
        tree = ET.parse(XML_FILE)
        root = tree.getroot()
        products = root.findall('product')
        print(f"[INIT] Найдено {len(products)} товаров")

        actions = []
        for prod in products:
            prod_id = prod.get('id')
            name = prod.findtext('name', '')
            category = prod.findtext('category', '')
            brand = prod.findtext('brand', '')
            keywords = prod.findtext('keywords', '')
            description = prod.findtext('description', '')

            price = float(prod.findtext('price', '0') or 0)
            package_size = int(prod.findtext('package_size', '1') or 1)

            weight_value = None
            weight_unit = ''
            weight_elem = prod.find('weight')
            if weight_elem is not None:
                try:
                    weight_value = float(weight_elem.text)
                    weight_unit = weight_elem.get('unit', '')
                except:
                    pass

            actions.append({
                "_index": INDEX_NAME,
                "_id": prod_id,
                "_source": {
                    "id": prod_id,
                    "name": name,
                    "category": category,
                    "brand": brand,
                    "keywords": keywords,
                    "description": description,
                    "weight_value": weight_value,
                    "weight_unit": weight_unit,
                    "price": price,
                    "package_size": package_size
                }
            })

        print(f"[INIT] Индексирую {len(actions)} документов...")
        success, failed = helpers.bulk(client, actions, chunk_size=100, raise_on_error=False)
        print(f"[INIT] Успешно загружено: {success}")
        if failed:
            print(f"[INIT] Ошибок: {len(failed)}")

        time.sleep(1)
        client.indices.refresh(index=INDEX_NAME)

        count = client.count(index=INDEX_NAME)['count']
        print(f"[INIT] В индексе {count} документов")

        if count == 0:
            print("[INIT] Индекс пуст!")
            return False

        return True

    except Exception as e:
        print(f"[INIT] Ошибка загрузки: {e}")
        import traceback
        traceback.print_exc()
        return False

def convert_layout(text: str) -> str:
    cyrillic_chars = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    latin_chars = sum(1 for c in text.lower() if 'a' <= c <= 'z')

    if cyrillic_chars > latin_chars:
        return ''.join(jcuken_to_qwerty.get(c, c) for c in text)
    else:
        return ''.join(qwerty_to_jcuken.get(c, c) for c in text)

def extract_numeric(query: str) -> Dict[str, Any]:
    patterns = [
        r'(\d+\.?\d*)\s*(л|l|литр)',
        r'(\d+\.?\d*)\s*(кг|kg|килограмм)',
        r'(\d+\.?\d*)\s*(г|g|грамм)',
        r'(\d+\.?\d*)\s*(мл|ml)',
    ]

    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            unit_map = {'л': 'l', 'литр': 'l', 'кг': 'kg', 'килограмм': 'kg', 'г': 'g', 'грамм': 'g', 'мл': 'ml'}
            normalized_unit = unit_map.get(unit, unit)
            clean_query = re.sub(pattern, '', query.lower()).strip()
            return {'value': value, 'unit': normalized_unit, 'clean_query': clean_query}

    return {'value': None, 'unit': None, 'clean_query': query}

def preprocess_query(query: str) -> List[str]:
    variants = [query.strip()]
    converted = convert_layout(query)
    if converted != query:
        variants.append(converted)
    normalized = ' '.join(query.split())
    if normalized not in variants:
        variants.append(normalized)
    return variants

def build_search_query(query_variants: List[str], numeric_attr: Dict) -> Dict:
    should_clauses = []
    for variant in query_variants:
        should_clauses.extend([
            {"match_bool_prefix": {"name.ngram": {"query": variant, "boost": 3.0}}},
            {"match_bool_prefix": {"keywords": {"query": variant, "boost": 2.5}}},
            {"match": {"brand": {"query": variant, "boost": 2.0, "fuzziness": "AUTO", "prefix_length": 1}}},
            {"match": {"category": {"query": variant, "boost": 1.5}}},
            {"match": {"description": {"query": variant, "boost": 1.0}}}
        ])

    query_body = {"query": {"bool": {"should": should_clauses, "minimum_should_match": 1}}}

    if numeric_attr['value'] is not None:
        query_body["query"]["bool"]["filter"] = [
            {"range": {"weight_value": {"gte": numeric_attr['value'] * 0.9, "lte": numeric_attr['value'] * 1.1}}}
        ]

    return query_body

def filter_results(results: List[Dict]) -> List[Dict]:
    if not results:
        return results

    max_score = results[0]['_score']
    threshold = max_score * 0.3
    filtered = [r for r in results if r['_score'] >= threshold]

    if len(filtered) >= 3:
        top3_categories = [r['_source']['category'] for r in filtered[:3]]
        category_counts = Counter(top3_categories)

        if category_counts.most_common(1)[0][1] >= 2:
            dominant_category = category_counts.most_common(1)[0][0]
            filtered_by_cat = [r for r in filtered if r['_source']['category'] == dominant_category]
            if len(filtered_by_cat) >= 3:
                return filtered_by_cat

    return filtered

@app.get("/health")
def health():
    try:
        count = client.count(index=INDEX_NAME)['count']
        return {"status": "healthy", "documents": count}
    except:
        return {"status": "unhealthy"}

@app.get("/search")
def search(q: str = Query(...), top_k: int = Query(5, ge=1, le=50)):
    start_time = time.time()
    try:
        numeric_attr = extract_numeric(q)
        query_variants = preprocess_query(q)

        search_body = build_search_query(query_variants, numeric_attr)
        search_body["size"] = top_k * 2

        response = client.search(index=INDEX_NAME, body=search_body)
        hits = response['hits']['hits']

        filtered_hits = filter_results(hits)[:top_k]

        results = []
        for hit in filtered_hits:
            source = hit['_source']
            results.append({
                "id": source['id'],
                "name": source['name'],
                "category": source['category'],
                "brand": source['brand'],
                "price": source['price'],
                "score": hit['_score']
            })

        latency = (time.time() - start_time) * 1000
        return {
            "query": q,
            "query_variants": query_variants,
            "numeric_filter": numeric_attr,
            "total": len(results),
            "results": results,
            "latency": round(latency, 1)
        }
    except Exception as e:
        return {"error": str(e), "query": q}

def evaluate():
    print("\n" + "="*50)
    print("EVALUATION")
    print("="*50)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
            queries = list(csv.DictReader(f))
        print(f"Загружено {len(queries)} запросов")
    except Exception as e:
        print(f"Ошибка загрузки запросов: {e}")
        return

    results = []
    successful = 0
    relevant = 0
    latencies = []

    for idx, row in enumerate(queries, 1):
        query = row.get('query', '')
        site = row.get('site', '')
        q_type = row.get('type', '')
        notes = row.get('notes', '')

        print(f"[{idx}/{len(queries)}] {query}")

        try:
            response = search(q=query, top_k=5)

            if 'error' not in response:
                successful += 1
                latencies.append(response.get('latency', 0))

                if response.get('total', 0) > 0:
                    relevant += 1
                    top_names = " | ".join([r['name'][:40] for r in response.get('results', [])[:3]])
                    top_categories = " | ".join([r['category'] for r in response.get('results', [])[:3]])
                    top_ids = " | ".join([r['id'] for r in response.get('results', [])[:3]])
                    top_scores = " | ".join([str(round(r['score'], 1)) for r in response.get('results', [])[:3]])
                else:
                    top_names = top_categories = top_ids = top_scores = ""

                results.append({
                    'query': query,
                    'site': site,
                    'type': q_type,
                    'notes': notes,
                    'status': 'ok',
                    'top_names': top_names,
                    'top_categories': top_categories,
                    'top_ids': top_ids,
                    'top_scores': top_scores,
                    'total_results': response.get('total', 0),
                    'latency_ms': response.get('latency', 0),
                    'error': ''
                })
            else:
                results.append({
                    'query': query,
                    'site': site,
                    'type': q_type,
                    'notes': notes,
                    'status': 'error',
                    'top_names': '',
                    'top_categories': '',
                    'top_ids': '',
                    'top_scores': '',
                    'total_results': 0,
                    'latency_ms': 0,
                    'error': response.get('error', '')
                })
        except Exception as e:
            results.append({
                'query': query,
                'site': site,
                'type': q_type,
                'notes': notes,
                'status': 'error',
                'top_names': '',
                'top_categories': '',
                'top_ids': '',
                'top_scores': '',
                'total_results': 0,
                'latency_ms': 0,
                'error': str(e)
            })

    csv_path = REPORTS_DIR / "evaluation_results.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    total = len(queries)
    success_rate = (successful / total * 100) if total > 0 else 0
    coverage = (relevant / total * 100) if total > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    metrics = {
        "total_queries": total,
        "successful_queries": successful,
        "relevant_queries": relevant,
        "success_rate": f"{success_rate:.1f}%",
        "coverage": f"{coverage:.1f}%",
        "avg_latency_ms": f"{avg_latency:.1f}"
    }

    json_path = REPORTS_DIR / "metrics.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ")
    print("="*50)
    print(f"total_queries: {total}")
    print(f"successful_queries: {successful}")
    print(f"relevant_queries: {relevant}")
    print(f"success_rate: {success_rate:.1f}%")
    print(f"coverage: {coverage:.1f}%")
    print(f"avg_latency_ms: {avg_latency:.1f}")
    print(f"Результаты: {csv_path}")
    print(f"Метрики: {json_path}")
    print("="*50 + "\n")

def main():
    print("\n" + "="*50)
    print("ПОЛНОЕ РЕШЕНИЕ")
    print("="*50)

    if not connect_opensearch():
        return

    delete_old_index()
    if not create_index():
        return

    if not load_documents():
        return

    print("\n" + "="*50)
    print("ИНИЦИАЛИЗАЦИЯ ЗАВЕРШЕНА")
    print("="*50)

    print("\nПРИМЕРЫ ПОИСКА:\n")

    test_queries = ["масло", "xfq", "йогурт греческий", "prosecco ro", "лампа led е27"]
    for test_q in test_queries:
        resp = search(q=test_q, top_k=3)
        print(f"  • '{test_q}': {resp.get('total', 0)} результатов")

    print("\nЗАПУСК EVALUATION...\n")
    evaluate()

    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="warning")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Остановка...")
        exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        exit(1)
