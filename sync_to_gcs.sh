gsutil -m rsync -r \
  -x '^(?!dataset_info\.json|iclr_|images).*' \
  data \
  gs://autoreviewer-data/autoreviewer_data
