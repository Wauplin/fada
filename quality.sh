#!/bin/bash

# Define an associative array where the key is a combination of builder name and config name,
# and the value is the model ID
declare -A datasets
datasets["glue:sst2"]="distilbert-base-uncased-finetuned-sst-2-english"
datasets["imdb:plain_text"]="textattack/bert-base-uncased-imdb"
# datasets["ag_news:default"]="textattack/bert-base-uncased-ag-news"
# datasets["yahoo_answers_topics:yahoo_answers_topics"]="fabriceyhc/bert-base-uncased-yahoo_answers_topics"
datasets["SetFit/sst5:default"]="SetFit/distilbert-base-uncased__sst5__all-train"
datasets["trec:default"]="aychang/bert-base-cased-trec-coarse"
datasets["yelp_polarity:plain_text"]="textattack/bert-base-uncased-yelp-polarity"
datasets["yelp_review_full:yelp_review_full"]="Ramamurthi/distilbert-base-uncased-finetuned-yelp-reviews"

# Iterate over the associative array
for dataset in "${!datasets[@]}"; do
    IFS=':' read -ra ADDR <<< "$dataset"
    builder_name=${ADDR[0]}
    config_name=${ADDR[1]}
    model_id=${datasets[$dataset]}

    # Execute the python command
    python -m eval.quality dataset.builder_name=$builder_name dataset.config_name=$config_name quality.model_id=$model_id
done
