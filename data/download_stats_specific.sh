#!/bin/bash
awscli="$SCRATCH/aws-cli/v2/2.11.25/dist/aws"
# alias awscli=aws


"$awscli" s3 sync --no-sign-request \
    --exclude='*' \
    --include aseg.stats \
    --include lh.BA.stats \
    --include lh.aparc.a2009s.stats \
    --include lh.aparc.stats \
    --include lh.entorhinal_exvivo.stats \
    --include rh.BA.stats \
    --include rh.aparc.a2009s.stats \
    --include rh.aparc.stats \
    --include rh.entorhinal_exvivo.stats \
    --include wmparc.stats \
    s3://fcp-indi/data/Projects/ABIDE/Outputs/freesurfer/5.1/ \
    ABIDE-I

