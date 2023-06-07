#!/bin/bash
alias awscli="$SCRATCH/aws-cli/v2/2.11.25/dist/aws"
alias awscli=aws


awscli s3 sync --no-sign-request \
  --exclude='*' \
  --include='*aseg.stats' \
  --include='*aparc.stats' \
  s3://fcp-indi/data/Projects/ABIDE/Outputs/freesurfer/