<<<<<<< HEAD
gcloud alpha storage rsync -r \
  --gzip-in-flight-all \
  -x '^(?!dataset_info\.json|iclr_|images/|2017_|2020_).*' \
=======
gsutil -m rsync -r \
  -x '^(?!dataset_info\.json|iclr_).*' \
>>>>>>> inference_scaling
  data \
  gs://autoreviewer-data/autoreviewer_data
