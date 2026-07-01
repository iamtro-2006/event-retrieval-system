from elasticsearch import Elasticsearch

es = Elasticsearch("http://192.168.1.15:9200")


print(es.info())

print(es.count(index="ocr"))