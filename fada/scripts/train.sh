python -m eval.train dataset.builder_name=glue dataset.config_name=sst2
python -m eval.train dataset.builder_name=imdb dataset.config_name=plain_text
python -m eval.train dataset.builder_name=ag_news dataset.config_name=default
python -m eval.train dataset.builder_name=yahoo_answers_topics dataset.config_name=yahoo_answers_topics