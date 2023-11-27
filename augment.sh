#!/bin/bash

# Define an associative array for builder_name and config_name pairs
declare -A datasets
datasets["glue"]="sst2"
datasets["imdb"]="plain_text"
datasets["ag_news"]="default"
datasets["yahoo_answers_topics"]="yahoo_answers_topics"
datasets["SetFit/sst5"]="default"
datasets["trec"]="default"
datasets["yelp_polarity"]="plain_text"
datasets["yelp_review_full"]="yelp_review_full"

# Array for num_per_class values
num_per_classes=(10 200 2500)

# Loop over each dataset
for builder in "${!datasets[@]}"; do
    config=${datasets[$builder]}
    for num in "${num_per_classes[@]}"; do
        # Run the command with the current dataset and num_per_class
        echo $builder $config $num
        python -m fada.fada_augment dataset.builder_name="$builder" dataset.config_name="$config" dataset.num_per_class="$num" augment.force=True
    done
done
