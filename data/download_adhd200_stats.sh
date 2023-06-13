#!/bin/bash
awscli="$SCRATCH/aws-cli/v2/2.11.25/dist/aws"
# alias awscli=aws


"$awscli" s3 sync --no-sign-request \
    --exclude='*' \
    --include '*.stats' \
    s3://fcp-indi/data/Projects/ADHD200/Outputs/fmriprep/freesurfer/ \
    ADHD-200

wget http://fcon_1000.projects.nitrc.org/indi/adhd200/general/ADHD-200_PhenotypicKey.pdf

echo "You will need to login to NITRC and manually download the file linked below:"
echo "https://www.nitrc.org/frs/download.php/9024/adhd200_preprocessed_phenotypics.tsv"
