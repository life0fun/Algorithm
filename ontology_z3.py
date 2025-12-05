from z3 import Solver, Function, Bool, Const, Consts, And, Implies, Not, sat, unsat, DeclareSort, BoolSort, ForAll

ontology = {
    # OWL Axioms(Rules). Implied conditional Property chain in 2+ properties for inference.
    # Format: (AxiomType, Arg1, Arg2, ...)
    "owl_rules": [
        ("Subsumption", "Device", "Transmitter"),
        ("Subsumption", "Transmitter", "PropertyX"), 
        ("Subsumption", "Transmitter", "NotPropertyX"),
        ("Subsumption", "MIMO", "AntennaArray"),
        ("FunctionalProperty", "hasVersion"),  # functional property can only have unique value.
        ("PropertyDomain", "transmits", "Transmitter"), # ForAll(E,Y) transmit(E,Y) implies Transmitter(E); 
        ("PropertyChain", "transmits", "encryptedBy", "UseEncryption")
    ],
    # RDF Triples (Facts). Obj
    # Format: (Subject, Predicate, Object)
    "rdf_facts": [
        ("T1", "type", "Transmitter"),
        ("T1", "use", "M1"),
        ("M1", "type", "MIMO"),
        ("T1", "hasVersion", "V1"), 
        ("T1", "hasVersion", "V2")
    ]
}

"""
SMT solver to find a solution that satisfies all constraints(un/conditional).
A constraint is a logic formula bool expr must be true in solution.
All RDF triples are modeled as predicate(HasProp(object, prop)) unconditionally true.
  solver.add(transmits(A,B))  
Axioms: if-then Rules, conditional constraints. If(Implies(A,B)), then(usesEncryption(B,A) true;
  solver.add(Implies(x==4, y>6)).
  solver.add(Implies(transmits(A, B), usesEncryption(A, B))) 
"""
def translate_ontology_to_z3(owl_axioms, rdf_triples):
    """
    Translates OWL Axioms to Z3 logic rules (constraints) 
    and RDF Triples to Z3 assertions (facts).
    """
    # 1. Define SMT/FOL Sorts (Types)
    # The universe of entities in the ontology
    Entity = DeclareSort('Entity')
    
    # Dictionaries to store Z3 functions for creating constraints(axioms) of entity instance; 
    Z3_Classes = {} 
    Z3_Properties = {}
    Z3_Consts = {} # symbolic constants for instance of entities
    
    def object_entity(TypeName):
        """Creates a Z3 Const (individual) of the Entity sort for an instance."""
        if TypeName not in Z3_Consts:
            Z3_Consts[TypeName] = Const(TypeName, Entity)
        return Z3_Consts[TypeName]
    
    def class_predicate(ClassName):
        """Creates a Z3 classification predicate (Uninterpreted Function) to indicate entity is a class."""
        if ClassName not in Z3_Classes:
            # A class is modeled as a unary predicate: Class(entity) -> Bool
            Z3_Classes[ClassName] = Function(ClassName, Entity, BoolSort())
        return Z3_Classes[ClassName]

    # unconditional binary predicate(bool expr) "Object has prop"
    def property_predicate(prop_name):
        """Creates a z3 binary predicate of prop_name to link object must have property. """
        if prop_name not in Z3_Properties:
            # An object property is modeled as a binary predicate: link two entites -> Bool
            Z3_Properties[prop_name] = Function(prop_name, Entity, Entity, BoolSort())
        return Z3_Properties[prop_name]

    solver = Solver()
    print("\n--- A. Translating RDF Triples to Z3 Assertions (Facts) ---")
    for subject, prop_type, prop_value in rdf_triples:  
        subject_entity = object_entity(subject) # subject name to Entity
        if prop_type == "type":  # Triple: (Individual, type, Class)
            isClass = class_predicate(prop_value)
            subject_type_assertion = isClass(subject_entity) # Z3 Assertion: Class(Individual)
            print(f"Adding Subject Type: {subject} is a {prop_value}")
            solver.add(subject_type_assertion)
        else:   # Triple: (Individual, Property, Individual)
            hasProp = property_predicate(prop_type)
            prop_value = object_entity(prop_value)
            
            # Z3 Assertion: Property(Subject, Object)
            object_prop_assertion = hasProp(subject_entity, prop_value)
            print(f"Adding Subject property: {subject} {prop_type} {prop_value}")
            solver.add(object_prop_assertion)

    print("--- B. Translating OWL Axioms to Z3 Constraints (Rules) ---")
    E, Y, Y1, Y2, A, B, C = Consts('E, Y, Y1, Y2, A, B, C', Entity) # universal variable for Z3 ForAll variable
    for axiom_type, *args in owl_axioms:
        if axiom_type == "Subsumption": # Axiom: Subclass(A) Subsumes Subclass(B)  (B -> A)
            A, B = args # condition rules: if entity is B, thenit must also be A;
            isClassA, isClassB = class_predicate(A), class_predicate(B)
            # FOL: ForAll E, B(E) Implies A(E)
            taxonomy_rule = ForAll([E], Implies(isClassB(E), isClassA(E)))
            print(f"Adding Conditional Rule: {B} -> {A}")
            solver.add(taxonomy_rule)

        elif axiom_type == "FunctionalProperty":  # at most one uniq prop value.
            prop_name = args[0]
            hasProp = property_predicate(prop_name)
            
            # FOL: ForAll E, ForAll Y1, ForAll Y2, (P(E, Y1) AND P(E, Y2)) Implies (Y1 = Y2)
            single_value_rule = ForAll([E, Y1, Y2], 
                                     Implies(And(hasProp(E, Y1), 
                                                 hasProp(E, Y2)), 
                                             Y1 == Y2))
            print(f"Adding Constraints: Functional Property {prop_name}")
            solver.add(single_value_rule)
    
        # ("PropertyDomain", "transmits", "Transmitter"), entity has transmits prop implies it is a Transmitter.
        elif axiom_type == "PropertyDomain":  # transmits(E,Y) imples Transmitter(E)
            prop_name, implied_class = args 
            hasProp, isClass = property_predicate(prop_name), class_predicate(implied_class)
            # FOL: ForAll (E,Y), P(E, Y) Implies A(E)
            prop_domain_rule = ForAll([E, Y], Implies(hasProp(E, Y), isClass(E)))
            print(f"Adding Rule: PropertyDomain for property {prop_name} is {isClass}")
            solver.add(prop_domain_rule)
        # FOL: ForAll (A, B, C), (transmits(A,B) AND encryptedBy(B,C)) IMPLIES usesEncryption(A, C)
        elif axiom_type == "PropertyChain":  
            prop1_name, prop2_name, implied_prop_name = args 
            hasProp1, hasProp2, hasImpliedProp = property_predicate(prop1_name), 
            property_predicate(prop2_name), property_predicate(implied_class)

            # FOL: ForAll (A, B, C), P1(A,B) AND P2(B,C) Implies A(E, C)
            rule = ForAll([A,B,C], Implies(And(hasProp1(A, B), hasProp2(B,C)), hasImpliedProp(A,C)))
            print(f"Adding Rule: PropertyChain for property {prop1_name} {prop2_name} is {hasImpliedProp}")
            solver.add(rule)

    return solver, Z3_Classes, Z3_Properties, Z3_Consts
    
def ontology_z3():
    solver, Z3_C, Z3_P, Z3_I = translate_ontology_to_z3(ontology["owl_rules"], 
                                                           ontology["rdf_facts"])
    # 4. Check Consistency
    print("\n--- C. Checking Consistency in Z3 ---")
    result = solver.check()
    print(f"Solver Check Result: {result}") 

    if result == unsat:
        print("\n**Inconsistency Found!** The facts violate the FunctionalProperty rule.")
        # The contradiction is caused by T1 having two versions, V1 and V2, 
        # but 'hasVersion' was declared as Functional.
    return solver

def check_property_chain_rule():
    Entity = DeclareSort('Entity')
    solver = ontology_z3()
    # device transmit payload, payload encrypted by Algo, implies device use Algo.
    Dev, Payload, Algo = Consts('Dev Payload Algo', Entity)
    transmits = Function('transmits', Entity, Entity, BoolSort())
    isEncryptedBy = Function('isEncryptedBy', Entity, Entity, BoolSort())
    usesEncryption = Function('usesEncryption', Entity, Entity, BoolSort())
    solver.add(transmits(Dev, Payload))        # Dev transmits Payload
    solver.add(isEncryptedBy(Payload, Algo)) # Payload isEncryptedBy Algo
    solver.push()
    solver.add(Not(usesEncryption(Dev, Algo)))
    result = solver.check()
    solver.pop()
    if result == unsat:
        print("Result: UNSAT. (The inferred fact is required to satisfy the Chain rule.)")

def check_entailment():  # PropertyChain Inference
    solver = Solver()

    hasPart = Function('hasPart', Entity, Entity, BoolSort())
    hasSubPart = Function('hasSubPart', Entity, Entity, BoolSort())
    hasDeepPart = Function('hasDeepPart', Entity, Entity, BoolSort())

    # Axiom (Property Chain)
    A, B, C = Consts('A B C', Entity)
    chain_axiom = ForAll([A, B, C],
        Implies(
            And(hasPart(A, B), hasSubPart(B, C)), # Hypothesis
            hasDeepPart(A, C)                     # Conclusion (P)
        )
    )
    solver.add(chain_axiom)

    # --- RDF Triples (Hypothesis Facts) ---
    # D1: Device, M1: Module, C1: Chip
    D1, M1, C1 = Consts('D1 M1 C1', Entity)
    solver.add(hasPart(D1, M1))       # Fact 1: D1 hasPart M1
    solver.add(hasSubPart(M1, C1))    # Fact 2: M1 hasSubPart C1

    # --- Entailment Check ---
    # Conclusion (P): The rule demands D1 must haveDeepPart C1.
    inferred_prop = hasDeepPart(D1, C1)

    # Check the negation: Assume the inferred fact is FALSE (NOT P)
    solver.push()
    solver.add(Not(inferred_prop)) # negation of inferred prop.

    print("\n--- Entailment Check ---")
    print(f"Checking if NOT {inferred_prop} leads to UNSAT...")
    entailment_result = solver.check()
    solver.pop()

    if entailment_result == unsat:
        print(f"Result: UNSAT.")
        print(f"Conclusion: The fact {inferred_prop} is logically **entailed** by the axioms and facts.")


# solver asserted Action params congestion subsum to network. Patent A's determines method overlaps it.
def check_overlap_subsumption(): # CongestionLevel is NetworkLoad
    # 1. Define General Sorts (Domain Types)
    # Sorts define the types of entities in our graph.
    Device = DeclareSort('Device')
    SystemLoad = DeclareSort('SystemLoad')   # The general category for all load types
    ActionStep = DeclareSort('ActionStep')   # The general category for all action steps

    # 2. Define Predicates for Classes/Subtypes
    # Classes are modeled as unary predicates (functions returning Bool) over the SystemLoad Sort.
    isCL = Function('is_CongestionLevel', SystemLoad, BoolSort())
    isNL = Function('is_NetworkLoad', SystemLoad, BoolSort()) # The supertype
    isDetermine = Function('is_StepDetermine', ActionStep, BoolSort())
    basedOn = Function('basedOn', ActionStep, SystemLoad, BoolSort())

    solver = Solver()
    # --- Add Axioms (Logical Hierarchy Rules to establish conditions to conclusions ---
    E = Const('E', SystemLoad)
    # Axiom 1 (Subsumption): CongestionLevel IS-A NetworkLoad
    axiom_cl_nl = ForAll([E], Implies(isCL(E), isNL(E)))
    solver.add(axiom_cl_nl)
    print(f"Added Axiom (CL Subsumption): {axiom_cl_nl}")

    # Patent A defines StepAction and action's input.
    WD_A = Const('WD_A', Device)
    Step_A = Const('Step_A', ActionStep)
    CL_Input = Const('CL_Input', SystemLoad)
    # Assert Patent A's Feature: P-1b uses CongestionLevel
    solver.add(isCL(CL_Input)) # add fact cl_input is cl. 
    solver.add(basedOn(Step_A, CL_Input)) # add fact step_A based on cl input

    # --- Overlap Detection Query (Entailment Check) ---
    # Query Goal: Does the system IMPLY that the input used by Patent A (CL_Input) 
    # is also a NetworkLoad (the supertype)?
    # Target Conclusion: isNL(CL_Input)
    negation_assertion = Not(isNL(CL_Input)) # ask z3 to prove isNL(CL_Input) via fail negation assertion.

    # Check for Entailment: If the constraints AND the negation of the conclusion lead to UNSAT, 
    # then the conclusion is logically entailed (implied).
    solver.push() # Save the current solver state
    solver.add(negation_assertion) # Assert the negation (e.g., "The input is NOT a NetworkLoad")

    print("--- Checking Entailment (Overlap) ---")
    check_result = solver.check()
    solver.pop() # Restore state

    print(f"Result of checking the negation: {check_result}")

    if check_result == unsat:
        print("\n✅ OVERLAP DETECTED: The specific relationship is subsumed by the general one.")
        print("The solver proves that the input MUST be a NetworkLoad.")
    else:
        print("\n❌ NO OVERLAP: The relationship is not guaranteed to be subsumed.")
