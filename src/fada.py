import numpy as np
import pandas as pd
import amrlib
import pickle
import os
from functools import partial

from transform import *
from eval_pipelines import *
from featurizers.amr import *
from utils import *

class FADA:
    def __init__(self, dataset, transforms, featurizers, eval_pipeline, task_name="sentiment", eval_dataset_size=1000):
        self.dataset = dataset
        self.dataset_config = (dataset.builder_name, dataset.config_name)
        self.dataset_name = self.dataset_config[0]
        if "glue" in self.dataset_config:
            self.dataset_name = self.dataset_config[1]
        self.transforms = sorted(transforms, key=lambda t: t.__name__)
        self.featurizers = sorted(featurizers, key=lambda f: f.__name__)
        self.eval_pipeline = eval_pipeline  
        self.task_name = task_name
        self.eval_dataset_size = eval_dataset_size
        
        self.graphs_save_path                     = "./fadata/amr_graphs.pkl"
        self.feature_matrix_save_path             = "./fadata/feature_matrix.csv"
        self.feature_datasets_path                = "./fadata/feature_datasets.pkl"
        self.transform_feature_analysis_save_path = "./fadata/transform_feature_analysis.csv"
        self.transform_feature_policy_save_path   = "./fadata/transform_feature_policy.csv"
        self.policy_probabilities_save_path       = "./fadata/policy_probabilities.csv"
        
        # initializations =============================================================
        
        # load precomputed data if available
        self.amr_model                  = None
        self.graphs                     = self.load_graphs()
        self.feature_matrix             = self.load_feature_matrix()
        self.feature_datasets           = self.load_feature_datasets()
        self.transform_feature_analysis = self.load_transform_feature_analysis()
        self.transform_feature_policy   = self.load_transform_feature_policy()
        self.policy_probabilities       = self.load_policy_probabilities()
    
    
    def load_amr_model(self, max_sent_len=128):
        self.amr_model = amrlib.load_stog_model(max_sent_len=max_sent_len)

    def save_graphs(self):
        with open(self.graphs_save_path, "wb") as f:
            pickle.dump(self.graphs, f)
            
    def load_graphs(self):
        if os.path.exists(self.graphs_save_path):
            with open(self.graphs_save_path, "rb") as f:
                return pickle.load(f)
        else:
            return None

    def save_feature_matrix(self):
        self.feature_matrix.to_csv(self.feature_matrix_save_path, index=False)
            
    def load_feature_matrix(self):
        if os.path.exists(self.feature_matrix_save_path):
            return pd.read_csv(self.feature_matrix_save_path)
        else:
            return None
        
    def save_feature_datasets(self):
        with open(self.feature_datasets_path, "wb") as f:
            pickle.dump(self.feature_datasets, f)
            
    def load_feature_datasets(self):
        if os.path.exists(self.feature_datasets_path):
            with open(self.feature_datasets_path, "rb") as f:
                return pickle.load(f)
        else:
            return None
        
    def save_transform_feature_analysis(self):
        self.transform_feature_analysis.to_csv(self.transform_feature_analysis_save_path, index=False)
            
    def load_transform_feature_analysis(self):
        if os.path.exists(self.transform_feature_analysis_save_path):
            return pd.read_csv(self.transform_feature_analysis_save_path)
        else:
            return None
        
    def save_transform_feature_policy(self):
        self.transform_feature_policy.to_csv(self.transform_feature_policy_save_path)
            
    def load_transform_feature_policy(self):
        if os.path.exists(self.transform_feature_policy_save_path):
            return pd.read_csv(self.transform_feature_policy_save_path, index_col=0)
        else:
            return None
    
    def save_policy_probabilities(self):
        self.policy_probabilities.to_csv(self.policy_probabilities_save_path, index=False)
            
    def load_policy_probabilities(self):
        if os.path.exists(self.policy_probabilities_save_path):
            return pd.read_csv(self.policy_probabilities_save_path)
        else:
            return None
        
    def initialize_transforms(self):
        self.transforms = [Transform(t, task_name=self.task_name) for t in self.transforms]   
        
    def text_to_amr(self):
        if self.amr_model is None:
            self.load_amr_model()
        if self.graphs is None:
            amr_penmans = self.amr_model.parse_sents(self.dataset['text'], add_metadata=False, disable_progress=False)
            amr_graphs  = [AMRGraph(p) for p in amr_penmans]
            self.graphs = amr_graphs
            self.save_graphs()
        return self.graphs
    
    def generate_feature_matrix(self):
        if self.feature_matrix is None:
            feature_matrix = []
            for g in self.graphs:
                feature_vector = []
                for f in self.featurizers:
                    feature_vector.append(f(g))
                feature_matrix.append(feature_vector)
            feature_matrix = np.array(feature_matrix, dtype=np.bool_)
            df = pd.DataFrame(feature_matrix, columns=[f.__name__ for f in self.featurizers])
            self.feature_matrix = df
            self.save_feature_matrix()
        return self.feature_matrix
        
    def locate_feature_idx(self, featurizer_id, top_n=3, has_feature=True):
    
        feature_column = self.feature_matrix[:, featurizer_id]
        if not has_feature:
            feature_column = np.invert(feature_column)
        targets = np.where(feature_column)[0]

        num_examples = len(targets)
        if num_examples == 0:
            print(f"0 (not {top_n}) \t : {self.featurizers[featurizer_id].__name__}.")
            return []
        elif num_examples < top_n:
            print(f"{num_examples} (not {top_n}) \t : {self.featurizers[featurizer_id].__name__}.")
            top_n = num_examples

        # trying to find texts that don't have a lot of other 
        # competing features which may confound our results
        competing_features = self.feature_matrix[targets].sum(axis=1)
        sorted_idx = np.argsort(competing_features)

        return targets[sorted_idx][:top_n]
    
    def construct_feature_datasets(self):
        if self.feature_datasets is None:
            feature_datasets = {}
            for i, featurizer in enumerate(self.featurizers):

                feature_present_idx = self.locate_feature_idx(i, self.eval_dataset_size, has_feature=True)
                feature_absent_idx  = self.locate_feature_idx(i, len(feature_present_idx), has_feature=False)

                feature_present_dataset = self.dataset.select(feature_present_idx)
                feature_absent_dataset  = self.dataset.select(feature_absent_idx)

                feature_datasets[featurizer.__name__] = {
                    "feature_present_idx"     : feature_present_idx,
                    "feature_absent_idx"      : feature_absent_idx,
                    "feature_present_dataset" : feature_present_dataset,
                    "feature_absent_dataset"  : feature_absent_dataset,
                }
            self.feature_datasets = feature_datasets
            self.save_feature_datasets()
        return self.feature_datasets
    
    def transform_feature_datasets(self, batch_size=10, keep_originals=False):
        if self.feature_datasets is None:
            
            self.initialize_transforms()
            
            for f_name, f_data in self.feature_datasets.items():
                transformed_datasets = {}
                for t in self.transforms:
                    aug = partial(augment_data, 
                                  transform=t, 
                                  keep_originals=keep_originals)

                    feature_present_aug_dataset = f_data["feature_present_dataset"].map(aug, batched=True, batch_size=batch_size)
                    feature_absent_aug_dataset  = f_data["feature_absent_dataset"].map(aug, batched=True, batch_size=batch_size)

                    transformed_datasets[t.__name__] = {
                        "feature_present_aug_dataset" : feature_present_aug_dataset,
                        "feature_absent_aug_dataset"  : feature_absent_aug_dataset
                    }

                self.feature_datasets[f_name]["transformed_datasets"] = transformed_datasets
                self.save_feature_datasets()
        return self.feature_datasets
    
    def conduct_transform_feature_analysis(self):
        
        if self.transform_feature_analysis is None:
        
            if isinstance(self.eval_pipeline, TestPipeline):
                self.eval_pipeline.find_model_for_dataset(self.dataset_name)

            eval_results = []
            for f_name, f_data in self.feature_datasets.items():

                if f_data["feature_present_dataset"].num_rows == 0:
                    continue

                orig_T_acc = self.eval_pipeline.evaluate(f_data["feature_present_dataset"])
                orig_F_acc = self.eval_pipeline.evaluate(f_data["feature_absent_dataset"])

                orig_out = {
                    "transform": "Identity",
                    "featurizer": f_name,
                    "num_samples": f_data["feature_present_dataset"].num_rows,
                    "T_orig_acc": orig_T_acc,
                    "F_orig_acc": orig_F_acc,
                    "T_tran_acc": orig_T_acc,
                    "F_tran_acc": orig_F_acc,
                    "pct_T_changed": 0,
                    "pct_F_changed": 0
                }

                eval_results.append(orig_out)

                for t_name, t_data in f_data["transformed_datasets"].items():
                    print(f_name, t_name)

                    tran_T_acc = self.eval_pipeline.evaluate(t_data["feature_present_aug_dataset"])
                    tran_F_acc = self.eval_pipeline.evaluate(t_data["feature_absent_aug_dataset"])

                    pct_T_changed = percent_dataset_changed(f_data["feature_present_dataset"], t_data["feature_present_aug_dataset"])
                    pct_F_changed = percent_dataset_changed(f_data["feature_absent_dataset"],  t_data["feature_absent_aug_dataset"])

                    tran_out = {
                        "transform": t_name,
                        "featurizer": f_name,
                        "num_samples": f_data["feature_present_dataset"].num_rows,
                        "T_orig_acc": orig_T_acc,
                        "F_orig_acc": orig_F_acc,
                        "T_tran_acc": tran_T_acc,
                        "F_tran_acc": tran_F_acc,
                        "pct_T_changed": pct_T_changed,
                        "pct_F_changed": pct_F_changed
                    }

                    eval_results.append(tran_out)

            df = pd.DataFrame(eval_results)
            df['T_diff'] = df['T_tran_acc'] - df['T_orig_acc']
            df['F_diff'] = df['F_tran_acc'] - df['F_orig_acc']
            df["diff_impact"] =  df['T_diff'] - df['F_diff']
            df['impact'] = ((df["T_tran_acc"] + df["F_tran_acc"]) / 2 ) - ((df["T_orig_acc"] + df["F_orig_acc"]) / 2 )
            df['pct_changed'] = (df["pct_T_changed"] + df["pct_F_changed"]) / 2 

            self.transform_feature_analysis = df
            self.save_transform_feature_analysis()
        
        return self.transform_feature_analysis
    
    def construct_transform_feature_policy(self, min_samples=10):
        
        if self.transform_feature_policy is None:
        
            df = self.transform_feature_analysis
            df = df[~df["transform"].isin(["Identity"])]
            # df = df[df['num_samples'] >= min_samples]

            # percent changed =========================================
            cols = ['transform', 'featurizer', 'pct_changed']
            df_changed = df[cols].pivot_table(values='pct_changed', 
                                           index='transform', 
                                           columns='featurizer', 
                                           aggfunc='mean')

            # absolute performance impact =============================
            cols = ['transform', 'featurizer', 'impact']
            df_impact = df[cols].pivot_table(values='impact', 
                                           index='transform', 
                                           columns='featurizer', 
                                           aggfunc='mean')
            df_impact = normalize_minmax(df_impact)

            # differential performance impact =========================
            cols = ['transform', 'featurizer', 'diff_impact']
            df_diff = df[cols].pivot_table(values='diff_impact', 
                                           index='transform', 
                                           columns='featurizer', 
                                           aggfunc='mean')
            df_diff = normalize_minmax(df_diff)
            df_diff = normalize_sum(df_diff)

            # aggregated policy accounting for all 3 dimensions =======
            df_final = normalize_sum(df_changed * df_diff * df_impact)

            self.transform_feature_policy = df_final
            self.save_transform_feature_policy()
        
        return self.transform_feature_policy 
    
    def implement_policy_probabilities(self):
        
        if self.policy_probabilities is None:
            policy = self.transform_feature_policy
            default_probability = policy.mean(axis=1)
            
            policy_probs = []
            for i in range(len(self.feature_matrix)):
                available_features = self.feature_matrix.iloc[i][self.feature_matrix.iloc[i]].index.values
                if len(available_features) == 0:
                    probs = default_probability
                else:
                    probs = policy[available_features].mean(axis=1)
                policy_probs.append(probs)
            df = pd.concat(policy_probs, axis=1).T
            self.policy_probabilities = df
            self.save_policy_probabilities()
            
        return self.policy_probabilities