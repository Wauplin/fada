# python -m eval.train train.dataset_matcher=glue.sst2.*.10.* dataset.builder_name=glue dataset.config_name=sst2 train.visible_cuda_devices=0
# python -m eval.train train.dataset_matcher=imdb.plain_text.*.10.* dataset.builder_name=imdb dataset.config_name=plain_text train.visible_cuda_devices=1
# python -m eval.train train.dataset_matcher=ag_news.default.*.10.* dataset.builder_name=ag_news dataset.config_name=default train.visible_cuda_devices=2
# python -m eval.train train.dataset_matcher=yahoo_answers_topics.yahoo_answers_topics.*.10.* dataset.builder_name=yahoo_answers_topics dataset.config_name=yahoo_answers_topics train.visible_cuda_devices=3

# python -m eval.train train.dataset_matcher=glue.sst2.backtranslate.10 dataset.builder_name=glue dataset.config_name=sst2 train.visible_cuda_devices=0 train.save_path=./eval/results/glue.sst2.backtranslate.10.csv
# python -m eval.train train.dataset_matcher=glue.sst2.context_cont.10 dataset.builder_name=glue dataset.config_name=sst2 train.visible_cuda_devices=0 train.save_path=./eval/results/glue.sst2.context_cont.10.csv
# python -m eval.train train.dataset_matcher=glue.sst2.context_ins.10 dataset.builder_name=glue dataset.config_name=sst2 train.visible_cuda_devices=0 train.save_path=./eval/results/glue.sst2.context_ins.10.csv
# python -m eval.train train.dataset_matcher=glue.sst2.context_sub.10 dataset.builder_name=glue dataset.config_name=sst2 train.visible_cuda_devices=0 train.save_path=./eval/results/glue.sst2.context_sub.10.csv

# python -m eval.train train.dataset_matcher=ag_news.default.backtranslate.10 dataset.builder_name=ag_news dataset.config_name=default train.visible_cuda_devices=0 train.save_path=./eval/results/ag_news.default.backtranslate.10.csv
# python -m eval.train train.dataset_matcher=ag_news.default.context_cont.10 dataset.builder_name=ag_news dataset.config_name=default train.visible_cuda_devices=0 train.save_path=./eval/results/ag_news.default.context_cont.10.csv
# python -m eval.train train.dataset_matcher=ag_news.default.context_ins.10 dataset.builder_name=ag_news dataset.config_name=default train.visible_cuda_devices=0 train.save_path=./eval/results/ag_news.default.context_ins.10.csv
# python -m eval.train train.dataset_matcher=ag_news.default.context_sub.10 dataset.builder_name=ag_news dataset.config_name=default train.visible_cuda_devices=0 train.save_path=./eval/results/ag_news.default.context_sub.10.csv

python -m eval.train train.dataset_matcher=imdb.plain_text.backtranslate.10 dataset.builder_name=imdb dataset.config_name=plain_text train.visible_cuda_devices=0 train.save_path=./eval/results/imdb.plain_text.backtranslate.10.csv
python -m eval.train train.dataset_matcher=imdb.plain_text.context_cont.10 dataset.builder_name=imdb dataset.config_name=plain_text train.visible_cuda_devices=0 train.save_path=./eval/results/imdb.plain_text.context_cont.10.csv
python -m eval.train train.dataset_matcher=imdb.plain_text.context_ins.10 dataset.builder_name=imdb dataset.config_name=plain_text train.visible_cuda_devices=0 train.save_path=./eval/results/imdb.plain_text.context_ins.10.csv
python -m eval.train train.dataset_matcher=imdb.plain_text.context_sub.10 dataset.builder_name=imdb dataset.config_name=plain_text train.visible_cuda_devices=0 train.save_path=./eval/results/imdb.plain_text.context_sub.10.csv

# python -m eval.train train.dataset_matcher=yahoo_answers_topics.yahoo_answers_topics.backtranslate.10 dataset.builder_name=yahoo_answers_topics dataset.config_name=yahoo_answers_topics train.visible_cuda_devices=0 train.save_path=./eval/results/yahoo_answers_topics.yahoo_answers_topics.backtranslate.10.csv
# python -m eval.train train.dataset_matcher=yahoo_answers_topics.yahoo_answers_topics.context_cont.10 dataset.builder_name=yahoo_answers_topics dataset.config_name=yahoo_answers_topics train.visible_cuda_devices=0 train.save_path=./eval/results/yahoo_answers_topics.yahoo_answers_topics.context_cont.10.csv
# python -m eval.train train.dataset_matcher=yahoo_answers_topics.yahoo_answers_topics.context_ins.10 dataset.builder_name=yahoo_answers_topics dataset.config_name=yahoo_answers_topics train.visible_cuda_devices=0 train.save_path=./eval/results/yahoo_answers_topics.yahoo_answers_topics.context_ins.10.csv
# python -m eval.train train.dataset_matcher=yahoo_answers_topics.yahoo_answers_topics.context_sub.10 dataset.builder_name=yahoo_answers_topics dataset.config_name=yahoo_answers_topics train.visible_cuda_devices=0 train.save_path=./eval/results/yahoo_answers_topics.yahoo_answers_topics.context_sub.10.csv