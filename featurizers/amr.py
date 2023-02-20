import penman
import amrlib
import numpy as np

class AMRFeatureExtractor:
    
    def __init__(self):
        self.featurizers = featurizers = [    
            contains_accompanier,
            contains_age,
            contains_beneficiary,
            contains_concession,
            contains_condition,
            contains_conjunctions,
            contains_consist_of,
            contains_coreferences,
            contains_degree,
            contains_destination,
            contains_direction,
            contains_domain,
            contains_duration,
            contains_example,
            contains_exlamation,
            contains_extent,
            contains_frequency,
            contains_imperative,
            contains_instrument,
            contains_interrogative_clause,
            contains_location,
            contains_manner,
            contains_medium,
            contains_mod,
            contains_mode,
            contains_name,
            contains_negation,
            contains_number,
            contains_ord,
            contains_part,
            contains_path,
            contains_polarity,
            contains_polite,
            contains_poss,
            contains_purpose,
            contains_quant,
            contains_question,
            contains_range,
            contains_scale,
            contains_source,
            contains_subevent,
            contains_time,
            contains_topic,
            contains_unit
        ]
        self.featurizers = sorted(featurizers, key=lambda f: f.__name__)
        self.amr_model   = None
        
    def load_amr_model(self, max_sent_len=128):
        self.amr_model = amrlib.load_stog_model(max_sent_len=max_sent_len)
        
    def text_to_amr(self, texts):
        if self.amr_model is None:
            self.load_amr_model()
        amr_penmans = self.amr_model.parse_sents(texts, add_metadata=False, disable_progress=True)
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
        

class AMRGraph:
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

# attributes =============================================================

def contains_imperative(g): return g.contains_attribute("imperative")
def contains_exlamation(g): return g.contains_attribute("expressive")
def contains_negation(g):   return g.contains_attribute("-")

# concepts ===============================================================

def contains_conjunctions(g):         return g.contains_concept(["and", "or", "contrast-01", "either", "neither"])
def contains_interrogative_clause(g): return g.contains_concept("truth-value")
def contains_question(g):             return g.contains_concept(["amr-unknown", "amr-choice"])

# roles ==================================================================

def contains_coreferences(g): return any(r for r in g.amr_text.split() if r in ['i', 'you', 'he', 'she', 'it', 'we', 'they'])
def contains_number(g):       return any(a for a in g.graph.attributes() if a.target.isnumeric())

def contains_accompanier(g):  return g.contains_role(':accompanier')
def contains_age(g):          return g.contains_role(':age')
def contains_beneficiary(g):  return g.contains_role(':beneficiary')
def contains_concession(g):   return g.contains_role(':concession')
def contains_condition(g):    return g.contains_role(':condition')
def contains_consist_of(g):   return any(r for r in g.amr_text.split() if r in [':consist-of'])
def contains_degree(g):       return g.contains_role(':degree')
def contains_destination(g):  return g.contains_role(':destination')
def contains_direction(g):    return g.contains_role(':direction')
def contains_domain(g):       return g.contains_role(':domain')
def contains_duration(g):     return g.contains_role(':duration')
def contains_example(g):      return g.contains_role(':example')
def contains_extent(g):       return g.contains_role(':extent')
def contains_frequency(g):    return g.contains_role(':frequency')
def contains_instrument(g):   return g.contains_role(':instrument')
# def contains_li(g):           return g.contains_role(':li')
def contains_location(g):     return g.contains_role(':location')
def contains_manner(g):       return g.contains_role(':manner')
def contains_medium(g):       return g.contains_role(':medium')
def contains_mod(g):          return g.contains_role(':mod')
def contains_mode(g):         return any(a for a in g.graph.attributes() if ":mode" in a.role)
def contains_name(g):         return g.contains_role(':name')
def contains_ord(g):          return g.contains_role(':ord')
def contains_part(g):         return g.contains_role(':part')
def contains_path(g):         return g.contains_role(':path')
def contains_polarity(g):     return g.contains_role(':polarity')
def contains_polite(g):       return any(r for r in g.amr_text.split() if r in [':polite'])
def contains_poss(g):         return g.contains_role(':poss')
def contains_purpose(g):      return g.contains_role(':purpose')
def contains_quant(g):        return g.contains_role(':quant')
def contains_range(g):        return g.contains_role(':range')
def contains_scale(g):        return g.contains_role(':scale')
def contains_source(g):       return g.contains_role(':source')
def contains_subevent(g):     return g.contains_role(':subevent')
def contains_time(g):         return g.contains_role(':time')
def contains_topic(g):        return g.contains_role(':topic')
def contains_unit(g):         return g.contains_role(':unit')
# def contains_value(g):        return g.contains_role(':value')
def contains_wiki(g):         return g.contains_role(':wiki')