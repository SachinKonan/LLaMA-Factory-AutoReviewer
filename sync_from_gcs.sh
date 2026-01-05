gcloud alpha storage rsync -r \
  --checksums-only \
  --gzip-in-flight-all \
  gs://autoreviewer-data/autoreviewer_data \
  data
