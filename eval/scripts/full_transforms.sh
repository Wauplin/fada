# ag_news 
python -m fada.fada_search dataset.builder_name=ag_news dataset.config_name=default transforms=all dataset.task_name=topic amr_extractor.max_sent_len=64 amr_extractor.batch_size=8 alignment_extractor.model_id=textattack/bert-base-uncased-ag-news

python -m fada.fada_augment dataset.builder_name=ag_news dataset.config_name=default dataset.task_name=sentiment augment.technique=all transforms=all dataset.num_per_class=10
python -m fada.fada_augment dataset.builder_name=ag_news dataset.config_name=default dataset.task_name=sentiment augment.technique=all transforms=all dataset.num_per_class=200
python -m fada.fada_augment dataset.builder_name=ag_news dataset.config_name=default dataset.task_name=sentiment augment.technique=all transforms=all dataset.num_per_class=2500

python -m eval.train dataset.builder_name=ag_news dataset.config_name=default

python -m eval.robustness dataset.builder_name=ag_news dataset.config_name=default