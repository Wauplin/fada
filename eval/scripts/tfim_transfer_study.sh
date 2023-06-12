# glue.sst2 --> others
python -m fada.fada_augment dataset.builder_name=glue \
                            dataset.config_name=sst2 \
                            dataset.task_name=sentiment \
                            augment.technique=transfer \
                            augment.tfim_dataset_builder_name=imdb \
                            augment.tfim_dataset_config_name=plain_text \
                            augment.save_dir=./fada/fadata/transfer_datasets \
                            dataset.num_per_class=10 

python -m fada.fada_augment dataset.builder_name=glue \
                            dataset.config_name=sst2 \
                            dataset.task_name=sentiment \
                            augment.technique=transfer \
                            augment.tfim_dataset_builder_name=ag_news \
                            augment.tfim_dataset_config_name=default \
                            augment.save_dir=./fada/fadata/transfer_datasets \
                            dataset.num_per_class=10 

python -m fada.fada_augment dataset.builder_name=glue \
                            dataset.config_name=sst2 \
                            dataset.task_name=sentiment \
                            augment.technique=transfer \
                            augment.tfim_dataset_builder_name=yahoo_answers_topics \
                            augment.tfim_dataset_config_name=yahoo_answers_topics \
                            augment.save_dir=./fada/fadata/transfer_datasets \
                            dataset.num_per_class=10 

python -m eval.train dataset_dir=./fada/fadata/transfer_datasets ^
                     dataset.builder_name=glue ^
                     dataset.config_name=sst2 ^
                     train.save_path=./eval/results/glue.sst2.training-transfer.csv

python -m eval.robustness model_matcher=./eval/pretrained/glue.sst2.*tfim*

# imdb.plain_text --> others
python -m fada.fada_augment dataset.builder_name=imdb \
                            dataset.config_name=plain_text \
                            dataset.task_name=sentiment \
                            augment.technique=transfer \
                            augment.tfim_dataset_builder_name=glue \
                            augment.tfim_dataset_config_name=sst2 \
                            augment.save_dir=./fada/fadata/transfer_datasets \
                            dataset.num_per_class=10 

python -m fada.fada_augment dataset.builder_name=imdb \
                            dataset.config_name=plain_text \
                            dataset.task_name=sentiment \
                            augment.technique=transfer \
                            augment.tfim_dataset_builder_name=ag_news \
                            augment.tfim_dataset_config_name=default \
                            augment.save_dir=./fada/fadata/transfer_datasets \
                            dataset.num_per_class=10 

python -m fada.fada_augment dataset.builder_name=imdb \
                            dataset.config_name=plain_text \
                            dataset.task_name=sentiment \
                            augment.technique=transfer \
                            augment.tfim_dataset_builder_name=yahoo_answers_topics \
                            augment.tfim_dataset_config_name=yahoo_answers_topics \
                            augment.save_dir=./fada/fadata/transfer_datasets \
                            dataset.num_per_class=10 

python -m eval.train dataset_dir=./fada/fadata/transfer_datasets \
                     dataset.builder_name=imdb \
                     dataset.config_name=plain_text \
                     train.save_path=./eval/results/imdb.plain_text.training-transfer.csv

python -m eval.robustness model_matcher=./eval/pretrained/imdb.plain_text.*tfim*

# ag_news.default --> others
python -m fada.fada_augment dataset.builder_name=ag_news \
                            dataset.config_name=default \
                            dataset.task_name=topic \
                            augment.technique=transfer \
                            augment.tfim_dataset_builder_name=yahoo_answers_topics \
                            augment.tfim_dataset_config_name=yahoo_answers_topics \
                            augment.save_dir=./fada/fadata/transfer_datasets \
                            dataset.num_per_class=10 

python -m fada.fada_augment dataset.builder_name=ag_news \
                            dataset.config_name=default \
                            dataset.task_name=topic \
                            augment.technique=transfer \
                            augment.tfim_dataset_builder_name=glue \
                            augment.tfim_dataset_config_name=sst2 \
                            augment.save_dir=./fada/fadata/transfer_datasets \
                            dataset.num_per_class=10 

python -m fada.fada_augment dataset.builder_name=ag_news \
                            dataset.config_name=default \
                            dataset.task_name=topic \
                            augment.technique=transfer \
                            augment.tfim_dataset_builder_name=imdb \
                            augment.tfim_dataset_config_name=plain_text \
                            augment.save_dir=./fada/fadata/transfer_datasets \
                            dataset.num_per_class=10 

python -m eval.train dataset_dir=./fada/fadata/transfer_datasets \
                     dataset.builder_name=ag_news \
                     dataset.config_name=default \
                     train.save_path=./eval/results/ag_news.default.training-transfer.csv

python -m eval.robustness model_matcher=./eval/pretrained/ag_news.default.*tfim*

# yahoo_answers_topics.yahoo_answers_topics --> others
python -m fada.fada_augment dataset.builder_name=yahoo_answers_topics \
                            dataset.config_name=yahoo_answers_topics \
                            dataset.task_name=topic \
                            augment.technique=transfer \
                            augment.tfim_dataset_builder_name=ag_news \
                            augment.tfim_dataset_config_name=default \
                            augment.save_dir=./fada/fadata/transfer_datasets \
                            dataset.num_per_class=10 

python -m fada.fada_augment dataset.builder_name=yahoo_answers_topics \
                            dataset.config_name=yahoo_answers_topics \
                            dataset.task_name=topic \
                            augment.technique=transfer \
                            augment.tfim_dataset_builder_name=glue \
                            augment.tfim_dataset_config_name=sst2 \
                            augment.save_dir=./fada/fadata/transfer_datasets \
                            dataset.num_per_class=10 

python -m fada.fada_augment dataset.builder_name=yahoo_answers_topics \
                            dataset.config_name=yahoo_answers_topics \
                            dataset.task_name=topic \
                            augment.technique=transfer \
                            augment.tfim_dataset_builder_name=imdb \
                            augment.tfim_dataset_config_name=plain_text \
                            augment.save_dir=./fada/fadata/transfer_datasets \
                            dataset.num_per_class=10 

python -m eval.train dataset_dir=./fada/fadata/transfer_datasets \
                     dataset.builder_name=yahoo_answers_topics \
                     dataset.config_name=yahoo_answers_topics \
                     train.save_path=./eval/results/yahoo_answers_topics.yahoo_answers_topics.training-transfer.csv

python -m eval.robustness model_matcher=./eval/pretrained/yahoo_answers_topics.yahoo_answers_topics.*tfim*

