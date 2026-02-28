# Master PhD Action Plan & Study Checklist (Year 1)
**Project Title:** Intelligence artificielle adaptative et raisonnement explicable pour les systèmes d'apprentissage automatique
**Candidate:** DhiaEddine Saidi
**Supervisor:** Pr. Haythem Ghazouani
**Start Date:** February 22, 2026

This is the definitive, all-in-one guide for Year 1. It merges your official deliverables (T1-T4), the weekly onboarding pipeline (S1-S12), the tool strategy, and the specific concepts/resources you must master coming from a Software Engineering background.

---

## 1. The Tool Setup Strategy

To succeed in a PhD, your knowledge must be accessible, searchable, and version-controlled.

### Zotero (For managing PDFs and Citations)
Zotero is your library. You should never read a loose PDF.
- **The Setup:**
  - [ ] Install Zotero and the **Zotero Connector** browser extension.
  - [ ] Install the **ZotFile plugin** (download the `.xpi` file from their site, go to Tools -> Add-ons in Zotero, and install it). Configure it to rename PDFs automatically to `Author_Year_Title` and save them to `thesis_project/docs/literature`.
  - [ ] Mirror the 4 folders we created (`01_foundations`, `02_datasets_and_benchmarks`, `03_neuro_symbolic`, `04_applications`) as "Collections" (folders) inside Zotero.
- **The Workflow:** When you find a paper on arXiv or Google Scholar, click the Zotero Connector extension. It will download the metadata and the PDF perfectly into your library.

### Obsidian (For your "Second Brain" & Notes)
Obsidian is your lab notebook. We use it instead of Notion/Word because it saves notes as local Markdown files (`.md`).
- **The Setup:**
  - [ ] Download Obsidian (`obsidian.md`).
  - [ ] Choose "Open folder as vault" and select your `thesis_project/docs` directory.
  - [ ] Create a folder inside for your notes (e.g., `thesis_project/docs/notes`).
- **The Workflow (The 4-Pass Reading Method):**
  When you read a paper, create a new note in Obsidian. 
  1. **Pass 1:** Read Title, Abstract, Intro, Conclusion. Establish the *Claim*.
  2. **Pass 2:** Look at Figures, Tables, Evaluation setup. Establish the *Evidence*.
  3. **Pass 3:** Read the Methodology. Establish the *How*.
  4. **Pass 4:** Write a 3-bullet summary in Obsidian: What problem did they solve? What was their innovation? How does it relate to your thesis? Use tags (like `#GNN`, `#RAVEN`).

---

## 2. Phase 0: The Core AI Foundations (Ongoing Prerequisites)
*Action: Check these off as you study them alongside your thesis work. Don't wait to master them all before starting Phase 1.*

- [ ] **Mathematics & Core Machine Learning**
  - [ ] Linear Algebra (Vectors, Matrices, Tensors). *(Resource: 3Blue1Brown "Essence of Linear Algebra" on YouTube)*
  - [ ] Calculus (Derivatives, Gradients). *(Resource: 3Blue1Brown "Essence of Calculus" on YouTube)*
  - [ ] Gradient Descent & Backpropagation. *(Resource: Andrej Karpathy "Neural Networks: Zero to Hero" on YouTube)*
- [ ] **Deep Learning Architectures**
  - [ ] Artificial Neural Networks (ANN) & Multi-Layer Perceptrons.
  - [ ] Convolutional Neural Networks (CNN) for image processing. *(Resource: CS231n Stanford Lectures on YouTube)*
- [ ] **Graph Theory & GNNs**
  - [ ] Graph basics (Nodes, Edges, Adjacency Matrices).
  - [ ] Graph Convolutional Networks (GCN) & Message Passing. *(Resource: CS224W Stanford Lectures on YouTube - First 3 videos)*
- [ ] **Symbolic Logic & Knowledge**
  - [ ] First-Order Logic basics.
  - [ ] Differentiable Logic concepts.

---

## 3. Phase 1: Onboarding & State of the Art (T1 : Feb - June)
**Goal:** Setup tools, understand the theoretical landscape, and deliver the SOTA report.
**Official Deliverable:** Rapport État de l'Art (Due: June 30)

### S1-S2 (Feb 22 – Mar 7): Tools & Environment
*Status: Partially Completed.*
- [x] Install Python (Anaconda), PyTorch, Jupyter Lab.
- [x] Create the `thesis_project` code architecture (`src`, `data`, `notebooks`).
- [ ] Initialize Git and create a GitHub repository.

### S3-S4 (Mar 8 – Mar 21): Foundational Reading (The 4-Pass Method)
*Action: Create an Obsidian note for each paper summarizing the Problem, Innovation, and Relevance.*
- [ ] Read: *The graph neural network model.pdf* (Focus on the introduction).
- [ ] Read: *RAVEN a dataset for relational and analogical visual reasoning.pdf* (Focus on input/output structure).
- [ ] Read: *Graph neural networks meet neural-symbolic computing.pdf* (Focus on the integration of logic).
- [ ] Read: *Neuro-Symbolic hierarchical rule induction.pdf*.

### S5-S6 (Mar 22 – Apr 4): Benchmark Exploration
- [ ] Download the official RAVEN dataset (from GitHub: WellyZhang/RAVEN).
- [ ] Write a Jupyter Notebook (`notebooks/01_explore_raven.ipynb`) to load, visualize the 3x3 matrices, and inspect the underlying rule annotations.

### S7-S10 (Apr 5 – May 2): PyTorch Geometric (PyG) & The First Baseline
- [ ] Complete the [official PyTorch Geometric (PyG) Introduction Tutorial](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html) on a fake graph.
- [ ] Reproduce a standard, logic-less CNN baseline on the RAVEN dataset to establish your starting metrics (Accuracy, Loss).

### S11-S12 (May 3 – May 16): The Architectural Foundation
- [ ] Code the **Visual Encoder** (to extract features from Raven image panels).
- [ ] Code the **Graph Constructor** (to turn the 9 panels into nodes/edges).
- [ ] Document this entire pipeline in a clean, commented Jupyter Notebook.

### T1 Wrapping (May 17 – June 30): The SOTA Report
- [ ] Expand reading to 15-20 papers (using Zotero Connector).
- [ ] Consolidate Obsidian notes into chapters.
- [ ] Draft and finalize the `stat_of_the_art.tex` document locally.
- [ ] **SUBMIT:** Rapport État de l'Art (June 30).

---

## 4. Phase 2: The Neuro-Symbolic Baseline (T2 : July - Sept)
**Goal:** Build a complete GNN system capable of solving RAVEN (without the custom logic yet).
**Official Deliverable:** Prototype Baseline + Rapport d'étape (Due: Sept 30) *Required for Year 2 acceptance.*

- [ ] Connect the Visual Encoder and Graph Constructor to a full PyTorch GNN (e.g., a GAT or GCN).
- [ ] Train the GNN to predict the missing Raven panel.
- [ ] Record baseline performance metrics and inference times.
- [ ] Study architectural patterns for Differentiable Logic Modules (e.g., Logic Tensor Networks).
- [ ] Write the trimestrial progress report.
- [ ] **SUBMIT:** Prototype Baseline Code & Rapport d'étape (Sept 30).

---

## 5. Phase 3: The Original Contribution (T3 : Oct - Nov)
**Goal:** Inject logic into the baseline, extract rules, and prove explainability.
**Official Deliverable:** Rapport Expérimental + Code (Due: Nov 30)

- [ ] Inject the logic module (e.g., constraints on arithmetic/spatial relations) into the GNN's loss function or reasoning layer.
- [ ] Write the code to extract the symbolic rules the model used to make its decision.
- [ ] Evaluate Explainability: Is the extracted rule accurate? Is it stable?
- [ ] Compare Neuro-Symbolic performance vs. the T2 Baseline.
- [ ] Start drafting the scientific paper (Target: NeSy or XAI workshops).
- [ ] **SUBMIT:** Experimental Report & Commented Code (Nov 30).

---

## 6. Phase 4: Reflection & Publication (T4 : Dec)
**Goal:** Finalize the year, document the journey, and publish.
**Official Deliverable:** Article soumis + Rapport Annuel (Due: Dec 31)

- [ ] (Optional) Test the model's generalization on the ARC or PGM datasets.
- [ ] Polish the codebase and build a small interactive demo (Jupyter or Streamlit).
- [ ] Present your findings at the Limtic laboratory seminar.
- [ ] **SUBMIT:** Finalize and submit the scientific article (Dec 31).
- [ ] **SUBMIT:** Deliver the Annual PhD Report (Dec 31).
