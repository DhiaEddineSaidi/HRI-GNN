
# Logic Types
This note provides a comprehensive overview of various logics, their core operators.

|              Logic               |                                          What It Does (Core Idea)                                          |                                                        Example                                                        | Core Operators / Symbols                                                                                                                                                                                            |
| :------------------------------: | :--------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
|   ==**Propositional Logic**==    |                    ==Deals with simple true/false statements and their combinations.==                     |                           =="If it is raining **(P)** , then the ground is wet **(Q)** ."==                           | ==- **∧** : Conjunction (AND)<br>- **∨** : Disjunction (OR)<br>- **¬** : Negation (NOT)<br>- **→** : Implication (IF...THEN)<br>- **↔** : Biconditional (IFF)==                                                     |
| ==**First-Order Logic (FOL)**==  |   ==Represents objects, their properties (predicates), and relationships. Quantifies over individuals.==   |               =="**For all** x, if x is a human **(Human(x))** , then x is mortal **(Mortal(x))** ."==                | ==- **∀** : Universal Quantifier (for all)<br>- **∃** : Existential Quantifier (there exists)<br>- **∧, ∨, ¬, →** : Standard connectives==                                                                          |
| ==**Second-Order Logic (SOL)**== |            ==Extends FOL by quantifying over properties and relations, not just individuals.==             |                 =="**There exists a property P** such that P holds for Socrates but not for Plato."==                 | ==- **∀** (over predicates) : For all properties<br>- **∃** (over predicates) : There exists a property<br>- **∧, ∨, ¬, →, ∀, ∃** : All FOL operators==                                                             |
| ==**Higher-Order Logic (HOL)**== | ==Generalizes SOL to any order. Functions and predicates can take functions and predicates as arguments.== |                          ==λ-calculus expressions, functions operating on other functions.==                          | ==- **λ** : Lambda Abstraction (function definition)<br>- **∀, ∃** : Quantifiers<br>- **:** : Type annotations==                                                                                                    |
|         **Modal Logic**          |                               Adds operators for necessity and possibility.                                |         "**It is necessary that** if it's raining, the ground is wet. **It is possible that** it is raining."         | - **□** : Necessity (Box/It is necessary that)<br>- **◇** : Possibility (Diamond/It is possible that)                                                                                                               |
|        **Temporal Logic**        |                         Adds operators to reason about time (sequence of states).                          |                  "If a request is made **(Req)** , **eventually** a grant will occur **(Grant)** ."                   | - **G (or □)** : Globally/Always<br>- **F (or ◇)** : Finally/Eventually<br>- **X** : Next<br>- **U** : Until                                                                                                        |
|       **Epistemic Logic**        |                                  Adds operators for knowledge and belief.                                  |            "Agent A **knows that** it is raining, but does **not know that** agent B knows it is raining."            | - **K<sub>a</sub>** : Agent a knows that<br>- **B<sub>a</sub>** : Agent a believes that                                                                                                                             |
|        **Deontic Logic**         |                        Adds operators for obligation, permission, and prohibition.                         |          "You are **obligated to** do your homework. Therefore, you are **permitted to** do your homework."           | - **O** : Obligatory (It is obligatory that)<br>- **P** : Permitted (It is permitted that)<br>- **F** : Forbidden (It is forbidden that)                                                                            |
|       ==**Fuzzy Logic**==        |                ==Handles degrees of truth (values between 0 and 1), not just true/false.==                 |                    =="If the temperature is **slightly hot**, turn the fan on **a little bit**."==                    | ==- **μ(x)** : Membership function (degree of membership)<br>- **Hedges** : Modifiers like *very*, *slightly*, *somewhat*<br>- **Fuzzy rules** : IF (is hot) THEN (fan high)==                                      |
|     **Probabilistic Logic**      |                      Combines logical rules with probabilities to handle uncertainty.                      |                                                 "**Most birds fly.**                                                  | **P(** fly(bird) **)** = 0.9. Tweety is a bird. Therefore, Tweety flies with 0.9 probability."                                                                                                                      |
|      **Description Logic**       |     A decidable fragment of FOL for representing concepts, roles, and individuals in a structured way.     |                   "**Man** is a **subclass of** **Human**. Socrates is an **instance of** **Man**."                   | - **⊑** : Subsumption (is a subclass of)<br>- **⊔** : Union (or)<br>- **⊓** : Intersection (and)<br>- **∃** : Existential Restriction (some)<br>- **∀** : Universal Restriction (only)                              |
|     **Intuitionistic Logic**     |                 Rejects the law of excluded middle (P ∨ ¬P). Requires constructive proof.                  | Proving "There exists an irrational number a,b such that a^b is rational" requires actually constructing the numbers. | - **∧, ∨, ¬, →** : Connectives with a constructive interpretation<br>- **≠** : Does **not** include the law of excluded middle as an axiom                                                                          |
|         **Linear Logic**         |                      Treats propositions as resources that must be used exactly once.                      |    "If I have **one dollar ($1)** , I can buy a candy bar. After buying it, the dollar is **consumed** and gone."     | - **⊗** : Tensor (multiplicative conjunction - "and" for resources)<br>- **⅋** : Par (multiplicative disjunction)<br>- **!** : Of course (allows unlimited reuse)<br>- **&** : With (additive conjunction - choice) |
# Real Logic 
Real logic is built on a first order language with a signature that contains: 
 - constants to represent objects, (Example : `alice` , `bob`)
 - functional symbols to capture operations on those objects (Example: `father_of(alice)`, `age_of(bob)`),
 - relational symbols or predicates to describe relationships and variables that can range over the domain, Example: `Human(alice)`, `Loves(alice, bob)` . 
Formulas in real logic can take truth values between 0 `zero`  and 1 `one` , in other words they can be **partially true**. 
This is achieved by adopting a **fuzzy semantics**. 
- In real logic we use the term **grounding** instead of **~~interpretation~~**.
- The idea is that **domains** (Example of Domain : `People = {alice, bob, carol}`) are represented concretely by tensors that is by numerical structures in the real field.
### Grounding 
In real logic grounding connects symbols directly to vectors or tensors that can be used in computation. Grounding essentially means replacing the variables of a formula with actual values. a process sometimes called also **instantiation**.
Example :  if we have a **Predicate** : `isRed(x)`  and we **Ground** it with a **specific object** which is an `apple` we can evaluate it value not (`True` or `False`) but wit ha `value` in $\in[0,1]$
Grounding allows LTN to combine logic with machine learning : The Symbolic structure is still there, but it is tied directly to real-valued data through grounding.
### Language and signature 
- **Constants** become fixed or learned embeddings
- **variables** are sequences of these individual embedding
- **Functions** perform arbitrary mathematical mappings, whether fixed or learned
- **Predicates** are implemented by functions return a real number in the range $[0,1]$, which can be viewed as a degree of truth of the corresponding predicate or as the output of a differentiable classifier. Example `is_cat(x)` evaluates the **likelihood** if x is a cat
- **Quantifiers** are modeled with `aggregators`  that are  `symmetric and continuous` functions representing fuzzy generalizations of the respective quantifiers in classical logic.
	- Generalize the classical notions of **AND** and **OR** with **T-norms** and **S-norms** which are families of functions whose mathematical properties are consistent with the intuitive interpretation of logical **AND** and logical **OR** :
		- **T-norms** :  Triangular norm is the fuzzy equivalent of logical AND $\land$ It takes two values between zero and one and combines them into another value in the same range
		- **S-norms**
### Real Logic Recap 
A **first-order language** consists of:
1. **Constants** – to represent objects
    - Example: `alice`, `bob`
2. **Functional symbols** – to capture operations on those objects
    - Example: `father_of(alice)`, `age_of(bob)`
3. **Relational symbols or Predicates** – to describe relations
    - Example: `Human(alice)`, `Loves(alice, bob)`
4. **Variables** – That can range over the domain
    - Example: `x`, `y`, `z`
5. **Quantifiers** – to reason about multiple objects (specify “all” or “some”)
    - ∀x (for all x), ∃x (exists x)
### How Real Logic extends This 
In **Real Logic**:
- Constants, functions, predicates remain the same
- **Truth values** of predicates are in **\[0,1\]** instead of {0,1}
- Logical formulas become **differentiable** → can be optimized in a neural network
### Example
#### Step 1: Define a domain
Domain: People = {alice, bob, carol}
#### Step 2: Define constants
alice, bob, carol
#### Step 3: Define functions
father_of(x) → returns the father of x  
age_of(x) → returns age of x
#### Step 4: Define predicates
Human(x) → is x human?  
Parent(x,y) → is x a parent of y?  
Adult(x) → is x adult?
#### Step 5: Write formulas
**Classical logic version:**
∀x: Human(x) → Adult(father_of(x)) 
	“The father of every human is an adult”
**Real Logic version:**
- Evaluate predicates in **\[0,1\]**
- If `Adult(father_of(alice)) = 0.8` → partially satisfied

In real logic we use the term grounding instead of interpretation.
The idea is that domains are represented concretely by tensors that is by numerical structures in the real field.