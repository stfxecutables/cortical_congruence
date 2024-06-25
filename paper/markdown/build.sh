#!/bin/bash
IN=congruence.md
BIB=congruence.bib
OUT=congruence.html
STYLE=mdstyle.html
CITESTYLE=apa.csl

# --citeproc: process citations in --bibliography
# --mathjax: include mathjax js
# --standalone: more compatible with mathjax than --self-contained
# --resource-path: for e.g. included images, paths are relative to here
# --metadata: additional pandoc args
# --metadata: link-citations: link citations in text to bottom refs
# --metadata: link-bibliography: turn DOI / links to working hrefs
# --metadata: pagetitle: for HTML title output
# -H: include verbatim in html header
# -s: make output a full document and not a document fragment
# https://pandoc.org/MANUAL.html#extension-citations for more info
# bib file should be BibLaTeX
# Note: make sure you have latest pandoc (e.g. 2.18+)

# Since pandoc-crossref uses the same citation syntax as citeproc,
# you have to run former before latter. For example:
# https://lierdakil.github.io/pandoc-crossref/#citeproc-and-pandoc-crossref

pandoc \
    --standalone \
    --mathjax \
    --resource-path . \
    --filter pandoc-crossref \
    --citeproc \
    --bibliography $BIB \
    --csl $CITESTYLE \
    --metadata link-citations=true \
    --metadata link-bibliography=true \
    --metadata lang=en-US \
    --metadata tblPrefix=Table \
    --metadata figPrefix=Figure \
    --metadata eqnPrefix=Equation \
    --metadata linkReferences=true \
    -f markdown+citations -t html $IN \
    --metadata pagetitle="Cortical Congruence" \
    -H $STYLE \
    -s -o $OUT
echo -n "Built at $(date): " &&\
if ! command -v readlink &> /dev/null;
then
    greadlink -f $OUT;
else
    readlink -f $OUT;
fi
