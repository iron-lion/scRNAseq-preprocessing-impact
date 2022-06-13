#!/bin/bash
mkdir -p ./RAW;
cd ./RAW;

# download data
wget https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-3321/E-MTAB-3321.processed.1.zip
unzip E-MTAB-3321.processed.1.zip

