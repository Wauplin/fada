# !/bin/sh python

python -m fada.fada_augment dataset.builder_name=glue dataset.config_name=sst2 dataset.task_name=sentiment augment.technique=all dataset.num_per_class=10
python -m fada.fada_augment dataset.builder_name=glue dataset.config_name=sst2 dataset.task_name=sentiment augment.technique=all dataset.num_per_class=200
python -m fada.fada_augment dataset.builder_name=glue dataset.config_name=sst2 dataset.task_name=sentiment augment.technique=all dataset.num_per_class=2500

python -m fada.fada_augment dataset.builder_name=imdb dataset.config_name=plain_text dataset.task_name=sentiment augment.technique=all dataset.num_per_class=10
python -m fada.fada_augment dataset.builder_name=imdb dataset.config_name=plain_text dataset.task_name=sentiment augment.technique=all dataset.num_per_class=200
python -m fada.fada_augment dataset.builder_name=imdb dataset.config_name=plain_text dataset.task_name=sentiment augment.technique=all dataset.num_per_class=2500

python -m fada.fada_augment dataset.builder_name=ag_news dataset.config_name=default dataset.task_name=topic augment.technique=all dataset.num_per_class=10
python -m fada.fada_augment dataset.builder_name=ag_news dataset.config_name=default dataset.task_name=topic augment.technique=all dataset.num_per_class=200
python -m fada.fada_augment dataset.builder_name=ag_news dataset.config_name=default dataset.task_name=topic augment.technique=all dataset.num_per_class=2500

python -m fada.fada_augment dataset.builder_name=yahoo_answers_topics dataset.config_name=yahoo_answers_topics dataset.task_name=topic augment.technique=all dataset.num_per_class=10
python -m fada.fada_augment dataset.builder_name=yahoo_answers_topics dataset.config_name=yahoo_answers_topics dataset.task_name=topic augment.technique=all dataset.num_per_class=200
python -m fada.fada_augment dataset.builder_name=yahoo_answers_topics dataset.config_name=yahoo_answers_topics dataset.task_name=topic augment.technique=all dataset.num_per_class=2500