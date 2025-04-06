1. Download

```
mkdir -p jfk_documents && tail -n +2 sources.csv | awk -F',' '{print $1}' | xargs -n 1 wget -P jfk_documents
```
