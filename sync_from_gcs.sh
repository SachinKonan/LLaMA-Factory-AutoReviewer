gcloud alpha storage rsync -r \
  --gzip-in-flight-all \
  --exclude=".*_newreduction\.png$" \
  gs://autoreviewer-data/autoreviewer_data \
  data
