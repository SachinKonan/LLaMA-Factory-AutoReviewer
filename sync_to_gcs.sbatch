#!/bin/bash
#SBATCH --job-name=sync_gcs
#SBATCH --output=logs/sync_to_gcs_%j.out
#SBATCH --error=logs/sync_to_gcs_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=20
#SBATCH --partition=all

./sync_to_gcs.sh
