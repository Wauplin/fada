python -m eval.quality dataset.builder_name=glue dataset.config_name=sst2 quality.model_id=distilbert-base-uncased-finetuned-sst-2-english
python -m eval.quality dataset.builder_name=imdb dataset.config_name=plain_text quality.model_id=textattack/bert-base-uncased-imdb
python -m eval.quality dataset.builder_name=ag_news dataset.config_name=default quality.model_id=textattack/bert-base-uncased-ag-news
python -m eval.quality dataset.builder_name=yahoo_answers_topics dataset.config_name=yahoo_answers_topics quality.model_id=fabriceyhc/bert-base-uncased-yahoo_answers_topics