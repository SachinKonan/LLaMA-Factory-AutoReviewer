gcloud alpha storage rsync -r \
  --gzip-in-flight-all \
  -x '^(?!dataset_info\.json|iclr_|images/|2017_|2020_).*' \
  data \
  gs://autoreviewer-data/autoreviewer_data
