python -m eval.train dataset.builder_name=glue \
                     dataset.config_name=sst2 \
                     train.dataset_matcher=glue.sst2.eda* \
                     train.save_path=./eval/results/glue.sst2.eda.train.csv

python -m eval.train dataset.builder_name=glue \
                     dataset.config_name=sst2 \
                     train.dataset_matcher=glue.sst2.checklist* \
                     train.save_path=./eval/results/glue.sst2.checklist.train.csv

python -m eval.train dataset.builder_name=glue \
                     dataset.config_name=sst2 \
                     train.dataset_matcher=glue.sst2.taa* \
                     train.save_path=./eval/results/glue.sst2.taa.train.csv

python -m eval.train dataset.builder_name=imdb \
                     dataset.config_name=plain_text \
                     train.dataset_matcher=imdb.plain_text.eda* \
                     train.save_path=./eval/results/imdb.plain_text.eda.train.csv

python -m eval.train dataset.builder_name=imdb \
                     dataset.config_name=plain_text \
                     train.dataset_matcher=imdb.plain_text.checklist* \
                     train.save_path=./eval/results/imdb.plain_text.checklist.train.csv

python -m eval.train dataset.builder_name=imdb \
                     dataset.config_name=plain_text \
                     train.dataset_matcher=imdb.plain_text.taa* \
                     train.save_path=./eval/results/imdb.plain_text.taa.train.csv

python -m eval.train dataset.builder_name=ag_news \
                     dataset.config_name=default \
                     train.dataset_matcher=ag_news.default.eda* \
                     train.save_path=./eval/results/ag_news.default.eda.train.csv

python -m eval.train dataset.builder_name=ag_news \
                     dataset.config_name=default \
                     train.dataset_matcher=ag_news.default.checklist* \
                     train.save_path=./eval/results/ag_news.default.checklist.train.csv

python -m eval.train dataset.builder_name=yahoo_answers_topics \
                     dataset.config_name=yahoo_answers_topics \
                     train.dataset_matcher=yahoo_answers_topics.yahoo_answers_topics.eda* \
                     train.save_path=./eval/results/yahoo_answers_topics.yahoo_answers_topics.eda.train.csv

python -m eval.train dataset.builder_name=yahoo_answers_topics \
                     dataset.config_name=yahoo_answers_topics \
                     train.dataset_matcher=yahoo_answers_topics.yahoo_answers_topics.checklist* \
                     train.save_path=./eval/results/yahoo_answers_topics.yahoo_answers_topics.checklist.train.csv
