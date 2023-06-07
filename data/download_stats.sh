#!/bin/bash
awscli="$SCRATCH/aws-cli/v2/2.11.25/dist/aws"
# alias awscli=aws


# "$awscli" s3 sync --no-sign-request \
#    --exclude='*' \
#    --include '*.stats' \
#    s3://fcp-indi/data/Projects/ABIDE/Outputs/freesurfer/5.1/ \
#    ABIDE-I

"$awscli" s3 sync --no-sign-request \
    --exclude='*' \
    --include '*.stats' \
    s3://fcp-indi/data/Projects/ABIDE2/Outputs/fmriprep/freesurfer/ \
    ABIDE-II

