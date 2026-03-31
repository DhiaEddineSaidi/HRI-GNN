# Fuzzy Logic – Core Rules

## Definition

**Fuzzy Logic** is a form of logic where truth values are **continuous between 0 and 1**, instead of only `true = 1` or `false = 0`.

Truth value:

```
Truth(A) ∈ [0,1]
```

Example:

|Statement|Truth|
|---|---|
|temperature is hot|0.8|
|customer is risky|0.6|
|loan is safe|0.2|

---

# Core Fuzzy Logic Operators

## 1. Negation (NOT)

Classical logic:

```
¬A
```

Fuzzy rule:

```
¬A = 1 − A
```

Example:

|A|¬A|
|---|---|
|0.9|0.1|
|0.5|0.5|
|0.2|0.8|

---

## 2. Conjunction (AND)

Classical logic:

```
A ∧ B
```

Common fuzzy rule:

```
A ∧ B = min(A, B)
```

Example:

|A|B|A ∧ B|
|---|---|---|
|0.8|0.6|0.6|
|0.3|0.7|0.3|

Alternative (product logic):

```
A ∧ B = A × B
```

---

## 3. Disjunction (OR)

Classical logic:

```
A ∨ B
```

Fuzzy rule:

```
A ∨ B = max(A, B)
```

Example:

|A|B|A ∨ B|
|---|---|---|
|0.8|0.6|0.8|
|0.3|0.7|0.7|

---

## 4. Implication (IF–THEN)

Logical rule:

```
A → B
```

Classical equivalence:

```
A → B ≡ ¬A ∨ B
```

Fuzzy version:

```
A → B = max(1 − A, B)
```

Example:

|A|B|Rule value|
|---|---|---|
|0.9|0.8|0.8|
|0.9|0.2|0.2|
|0.2|0.3|0.8|

Interpretation:

- If **A is false**, the rule is satisfied.
    
- If **A is true**, **B must be true**.
    

---

# Summary Table

|Logical Operator|Fuzzy Formula|
|---|---|
|NOT|`¬A = 1 − A`|
|AND|`min(A,B)`|
|OR|`max(A,B)`|
|IMPLIES|`max(1 − A, B)`|

---

# Example (Risk Rule)

Rule:

```
IF income_high AND debt_low THEN loan_safe
```

Suppose:

```
income_high = 0.7
debt_low = 0.6
```

Compute:

```
income_high AND debt_low = min(0.7,0.6) = 0.6
```

So the rule truth value is **0.6**.

---

# Why Fuzzy Logic is Used in AI

Fuzzy logic allows:

- reasoning with **uncertain concepts**
    
- combining **logical rules with probabilities**
    
- integration with **neural networks**
    

This is why it is used in **neuro-symbolic AI systems**.