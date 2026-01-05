gcloud alpha storage rsync -r \
  --checksums-only \
  --gzip-in-flight-all \
  -x '^(?!dataset_info\.json|iclr_|images/).*' \
  data \
  gs://autoreviewer-data/autoreviewer_data
