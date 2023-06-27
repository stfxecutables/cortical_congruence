#!/bin/bash
awscli="$SCRATCH/aws-cli/v2/2.11.25/dist/aws"
# alias awscli=aws

echo "=============================================================================="
echo "Downlading ABIDE-I Data. Read notes below."
echo "=============================================================================="


"$awscli" s3 sync --no-sign-request \
    --exclude='*' \
    --include '*.stats' \
    s3://fcp-indi/data/Projects/ABIDE/Outputs/freesurfer/5.1/ \
    ABIDE-I

echo "=============================================================================="
echo "Downlading ABIDE-II Data. Read notes below."
echo "=============================================================================="

"$awscli" s3 sync --no-sign-request \
    --exclude='*' \
    --include '*.stats' \
    s3://fcp-indi/data/Projects/ABIDE2/Outputs/fmriprep/freesurfer/ \
    ABIDE-II


"$awscli" s3 sync --no-sign-request \
    --exclude='*' \
    --include 'Phenotypic_V1_0b.csv' \
    --include 'Phenotypic_V1_0b_preprocessed.csv' \
    --include 'Phenotypic_V1_0b_preprocessed1.csv' \
    s3://fcp-indi/data/Projects/ABIDE_Initiative/ \
    ABIDE-II

"$awscli" s3 cp --no-sign-request \
    s3://fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b.csv \
    ABIDE-II

"$awscli" s3 cp --no-sign-request \
    s3://fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed.csv \
    ABIDE-II


"$awscli" s3 sync --no-sign-request \
    s3://fcp-indi/data/Projects/ABIDE_Initiative/Resources/ \
    ABIDE-II/resources

"$awscli" s3 sync --no-sign-request \
    s3://fcp-indi/data/Projects/ABIDE_Initiative/PhenotypicData/ \
    ABIDE-II/phenotypic_data

wget http://fcon_1000.projects.nitrc.org/indi/abide/ABIDEII_Data_Legend.pdf

echo "See http://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html"
echo "for links to download ABIDE-II phenotypic data:"
echo ""
echo "    https://www.nitrc.org/frs/downloadlink.php/9108"
echo "    https://www.nitrc.org/frs/downloadlink.php/9109"
echo ""

echo ""
echo "=============================================================================="
echo "Downlading QTIM Data. Read notes below."
echo "=============================================================================="
echo ""

"$awscli" s3 cp --no-sign-request \
    s3://openneuro.org/ds004169/ \
    QTIM/phenotypic_data


"$awscli" s3 sync --no-sign-request \
    --exclude='*' \
    --include '*.tsv' \
    s3://openneuro.org/ds004169/derivatives \
    QTIM/

echo ""
echo "=============================================================================="
echo "Downlading QTAB Data. Read notes below. Note there FS stats are in QTIM."
echo "=============================================================================="
echo ""

"$awscli" s3 cp --no-sign-request \
    s3://openneuro.org/ds004146/participants.tsv \
    QTAB/phenotypic_data/phenotypic_data.tsv

echo "Visit https://zenodo.org/record/7765506/accessrequest and request access "
echo "to the extended phenotypic data."

echo ""
echo "=============================================================================="
echo "Downlading HBN Data. Read notes below."
echo "=============================================================================="
echo ""

mkdir -p HBN/phenotypic_data
wget http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/File/_pheno/HBN_R10_Pheno.csv -O HBN/phenotypic_data/HBN_R10_Pheno.csv

"$awscli" s3 sync --no-sign-request \
    --exclude='*' \
    --include '*.stats' \
    s3://fcp-indi/data/Projects/HBN/derivatives/Freesurfer_version6.0.0/ \
    HBN/derivatives