<<<<<<< HEAD
gcloud alpha storage rsync -r \
  --gzip-in-flight-all \
  gs://autoreviewer-data/autoreviewer_data \
  data
=======
gsutil -m rsync -r gs://autoreviewer-data/autoreviewer_data data
>>>>>>> inference_scaling
