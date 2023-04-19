import penman
import amrlib
import numpy as np
import torch

class AMRGraph:
    """
    Wraps an AMR graph and enables parsing for different nodes, 
    edges, and attributes.

    Parameters
    ----------
    amr : list[str | penman.graph.Graph]
        The amr string or graph to be parsed.
    """
    def __init__(self, amr):
        self.graph = penman.decode(amr) if not isinstance(amr, penman.graph.Graph) else amr
        self.amr_text = penman.encode(self.graph)

    def contains_concept(self, concepts):
        """
        Concepts are nodes / instances in the AMR graph.
        """
        if not isinstance(concepts, list): concepts = [concepts]
        graph_concepts = [t.target for t in self.graph.instances()]
        return any(c for c in graph_concepts if c in concepts)

    def contains_role(self, roles):
        """
        Roles are edges in the AMR graph.
        """
        if not isinstance(roles, list): roles = [roles]
        graph_roles = [e.role for e in self.graph.edges()]
        return any(r for r in graph_roles if r in roles)

    def contains_attribute(self, attributes):
        """
        Attributes are properties of concept nodes, i.e. relationships to 
        constant values.
        """
        if not isinstance(attributes, list): attributes = [attributes]
        graph_attrs = [a.target for a in self.graph.attributes()]
        return any(a for a in graph_attrs if a in attributes)


class AMRFeatureExtractor:
    """
    Assists with converting raw text into AMR graphs, which are then
    parsed into boolean features (e.g. `contains_coreferences`) 
    indicating the presence or absence of that particular feature. 

    As of 2023.03.20, there are a fixed number of AMR featurizer 
    functions, ordered alphabetically so that index indicates
    features. 

    Requires:
        `pip install amrlib`
    """
    
    def __init__(self):
        self.featurizers = featurizers = [    
            self.contains_accompanier,
            self.contains_age,
            self.contains_beneficiary,
            self.contains_concession,
            self.contains_condition,
            self.contains_conjunctions,
            self.contains_consist_of,
            self.contains_coreferences,
            self.contains_degree,
            self.contains_destination,
            self.contains_direction,
            self.contains_domain,
            self.contains_duration,
            self.contains_example,
            self.contains_exlamation,
            self.contains_extent,
            self.contains_frequency,
            self.contains_imperative,
            self.contains_instrument,
            self.contains_interrogative_clause,
            self.contains_location,
            self.contains_manner,
            self.contains_medium,
            self.contains_mod,
            self.contains_mode,
            self.contains_name,
            self.contains_negation,
            self.contains_number,
            self.contains_ord,
            self.contains_part,
            self.contains_path,
            self.contains_polarity,
            self.contains_polite,
            self.contains_poss,
            self.contains_purpose,
            self.contains_quant,
            self.contains_question,
            self.contains_range,
            self.contains_scale,
            self.contains_source,
            self.contains_subevent,
            self.contains_time,
            self.contains_topic,
            self.contains_unit
        ]
        self.featurizers = sorted(featurizers, key=lambda f: f.__name__)
        self.device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.amr_model   = None
        
    def load_amr_model(self, max_sent_len=128):
        self.amr_model = amrlib.load_stog_model(max_sent_len=max_sent_len, batch_size=4, device=self.device)
        
    def text_to_amr(self, texts):
        if self.amr_model is None:
            self.load_amr_model()
        amr_penmans = self.amr_model.parse_sents(texts, add_metadata=False, disable_progress=False)
        amr_graphs = []
        for p in amr_penmans:
            try:
                amr_graphs.append(AMRGraph(p))
            except Exception as e: 
                print(e)
                print(p)
                amr_graphs.append(AMRGraph(p))
        return amr_graphs
    
    def generate_feature_matrix(self, graphs):
        feature_matrix = []
        for g in graphs:
            feature_vector = []
            for f in self.featurizers:
                feature_vector.append(f(g))
            feature_matrix.append(feature_vector)
        feature_matrix = np.array(feature_matrix, dtype=np.int32)
        return feature_matrix
    
    def __call__(self, texts):
        graphs = self.text_to_amr(texts)
        return self.generate_feature_matrix(graphs)

    # attributes =============================================================

    def contains_imperative(self, g): return g.contains_attribute("imperative")
    def contains_exlamation(self, g): return g.contains_attribute("expressive")
    def contains_negation(self, g):   return g.contains_attribute("-")

    # concepts ===============================================================

    def contains_conjunctions(self, g):         return g.contains_concept(["and", "or", "contrast-01", "either", "neither"])
    def contains_interrogative_clause(self, g): return g.contains_concept("truth-value")
    def contains_question(self, g):             return g.contains_concept(["amr-unknown", "amr-choice"])

    # roles ==================================================================

    def contains_coreferences(self, g): return any(r for r in g.amr_text.split() if r in ['i', 'you', 'he', 'she', 'it', 'we', 'they'])
    def contains_number(self, g):       return any(a for a in g.graph.attributes() if a.target.isnumeric())

    def contains_accompanier(self, g):  return g.contains_role(':accompanier')
    def contains_age(self, g):          return g.contains_role(':age')
    def contains_beneficiary(self, g):  return g.contains_role(':beneficiary')
    def contains_concession(self, g):   return g.contains_role(':concession')
    def contains_condition(self, g):    return g.contains_role(':condition')
    def contains_consist_of(self, g):   return any(r for r in g.amr_text.split() if r in [':consist-of'])
    def contains_degree(self, g):       return g.contains_role(':degree')
    def contains_destination(self, g):  return g.contains_role(':destination')
    def contains_direction(self, g):    return g.contains_role(':direction')
    def contains_domain(self, g):       return g.contains_role(':domain')
    def contains_duration(self, g):     return g.contains_role(':duration')
    def contains_example(self, g):      return g.contains_role(':example')
    def contains_extent(self, g):       return g.contains_role(':extent')
    def contains_frequency(self, g):    return g.contains_role(':frequency')
    def contains_instrument(self, g):   return g.contains_role(':instrument')
    # def contains_li(self, g):           return g.contains_role(':li') # does not appear to be supported by `amrlib`
    def contains_location(self, g):     return g.contains_role(':location')
    def contains_manner(self, g):       return g.contains_role(':manner')
    def contains_medium(self, g):       return g.contains_role(':medium')
    def contains_mod(self, g):          return g.contains_role(':mod')
    def contains_mode(self, g):         return any(a for a in g.graph.attributes() if ":mode" in a.role)
    def contains_name(self, g):         return g.contains_role(':name')
    def contains_ord(self, g):          return g.contains_role(':ord')
    def contains_part(self, g):         return g.contains_role(':part')
    def contains_path(self, g):         return g.contains_role(':path')
    def contains_polarity(self, g):     return g.contains_role(':polarity')
    def contains_polite(self, g):       return any(r for r in g.amr_text.split() if r in [':polite'])
    def contains_poss(self, g):         return g.contains_role(':poss')
    def contains_purpose(self, g):      return g.contains_role(':purpose')
    def contains_quant(self, g):        return g.contains_role(':quant')
    def contains_range(self, g):        return g.contains_role(':range')
    def contains_scale(self, g):        return g.contains_role(':scale')
    def contains_source(self, g):       return g.contains_role(':source')
    def contains_subevent(self, g):     return g.contains_role(':subevent')
    def contains_time(self, g):         return g.contains_role(':time')
    def contains_topic(self, g):        return g.contains_role(':topic')
    def contains_unit(self, g):         return g.contains_role(':unit')
    # def contains_value(self, g):        return g.contains_role(':value') # does not appear to be supported by `amrlib`
    def contains_wiki(self, g):         return g.contains_role(':wiki')
       

if __name__ == '__main__':

    texts = [
        "John and Mary went to six store.",
        "Does John like pizza?",
        "She loves him."
    ]

    fe = AMRFeatureExtractor()
    features = fe(texts)

    for t, f in zip(texts, features):
        print(f"{t} [features]: ({', '.join([fe.featurizers[i].__name__ for i, v in enumerate(f) if v == 1])})")