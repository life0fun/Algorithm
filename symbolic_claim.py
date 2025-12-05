# claim checker using symbolic computation
from z3 import Solver, Bool, And, Implies, Not, sat, unsat
import json
import networkx as nx

# define extracted ontology from claims.
ontology = {
  "components": [
    "Transmitter", "Receiver", "OFDM", "MIMO", "PowerControl", "Modulation_OFDM",
    "Encryption", "KeyManagement", "Channel", "Bandwidth", "Frequency",
    "AntennaArray", "FrameStructure", "ErrorCorrection", "Retransmission",
    "Throughput", "HighThroughputGoal", "IncreasedLatency", 
    "RegulatoryCompliance", "PowerLimit"
  ],
  # expert curated rules as Axiom/Implies Missing rules can affect completeness check.
  "rules": [
    {"if": ["Transmitter", "Modulation_OFDM"], "then": "FrameStructure"},
    {"if": ["Bandwidth","HighThroughputGoal"], "then": "MIMO"},
    {"if": ["MIMO"], "then": "AntennaArray"},
    {"if": ["AntennaArray"], "then": "PowerControl"},
    {"if": ["Encryption"], "then": "KeyManagement"},
    {"if": ["PowerControl"], "then": "PowerLimit"},
    {"if": ["ErrorCorrection"], "then": "Retransmission"},
    {"if": ["Retransmission"], "then": "IncreasedLatency"}
  ]
}

Entity = DeclareSort('Entity')

# Solver is to find a solution that satisfies all constraints.
# A constraint is a logic formula bool expr must be true in solution.
# A class can be modeled as an unary predicate: Class(entity) -> Bool
# Function(name, Entity, BoolSort())
def object_entity_consts(components):
    return {name: Const(name, Entity) for name in components}

def build_solver(entities, constraints):
    s = Solver()
    # adding conditional constraints/axioms(A=>B), Not asserting the truth of the variables;
    for conditional_rule in constraints:
        conds = [entities[c] for c in conditional_rule["if"] if c in entities]
        if len(conds) == 1:
            implication = Implies(conds[0], entities[conditional_rule["then"]])
        else:
            implication = Implies(And(*conds), entities[conditional_rule["then"]])
        s.add(implication)
    return s

# proof by contradiction, AKA entailment checking of a statement
def is_implied_true(solver, statement):
    checker = Solver()  # clone a solve to add Not(statement)
    checker.add(*solver.assertions())
    checker.add(Not(statement))
    return checker.check() == unsat

# check Implies(P, Q), for each fact in P, assert it.
def check_rule(solver, fact_assertions, rule_to_check):
    checker = Solver()
    checker.add(*solver.assertions())
    # for each fact in P, assert it true to so implied Q can be checked.
    for constraint in fact_assertions:
        checker.add(constraint) # assert constraint true to solver. 
    
    # entailment check
    if is_implied_true(checker, rule_to_check):
        return f"{rule_to_check} Implied True"
    
    checker.add(rule_to_check) # assert 
    result = checker.check() 
    if result == unsat: 
        return f"{rule_to_check} Implied False" 
    elif result == sat: 
        return f"{rule_to_check} Consistent (Not Forced)" 
    else: 
        return f"{rule_to_check} Unknown/Undecided"

def testRules(solver, entities):
    # No transitive rules links transmitter to power, hence rule not forced; 
    if_then_rule = Implies(entities["Transmitter"], entities["PowerLimit"])
    status = check_rule(solver, [entities["Transmitter"]], if_then_rule)
    print(f"Rule Transmitter -> PowerLimit: {status}") # not forced;
    # there are transitive rules link mimo to power, hence the rule is Implied true; 
    if_then_rule = Implies(entities["MIMO"], entities["PowerLimit"])
    status = check_rule(solver, [entities["MIMO"]], if_then_rule)
    print(f"Rule MIMO -> PowerLimit is: {status}") 

    # claim error correction won't lead to increased latency, claim is Implied False;
    if_then_rule = Implies(entities["ErrorCorrection"], Not(entities["IncreasedLatency"]))
    status = check_rule(solver, [entities["ErrorCorrection"]], if_then_rule)
    print(f"Rule ErrorCorrection -> Not(IncreasedLatency) is: {status}")

def testGraphTraversal(ontology):
    G = nx.DiGraph()
    for rule in ontology["rules"]:
        for antecedent in rule["if"]:
            G.add_edge(antecedent, rule["then"], label="implies")

    # Start from claimed aspect as root
    root = ["Transmitter","Modulation_OFDM","MIMO","Encryption", "ErrorCorrection"]
    reachable = set()
    for s_node in root:
        if s_node in G:
            reachable |= set(nx.descendants(G, s_node))
    print("\nDerived by graph traversal (descendants):", reachable)

if __name__ == "__main__":
    # map ontology components to boolean atom entities
    entities = object_entity_consts(ontology["components"])
    solver = build_solver(entities, ontology["rules"])
    # component_class(solver, entities)
    testRules(solver, entities)
    testGraphTraversal(ontology)
