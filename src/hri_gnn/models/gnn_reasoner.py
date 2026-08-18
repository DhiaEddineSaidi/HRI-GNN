"""
GNN Raisonneur pour le pipeline Neuro-Symbolique (CNN → GNN → HRI).

Ce module construit un graphe à partir des embeddings CNN des panels RAVEN
et utilise un Graph Attention Network (GAT) pour propager l'information
relationnelle entre panels voisins.

Rôle dans le pipeline :
    16 embeddings (16, 256)  ──►  GNN  ──►  8 scores (un par candidat)

Structure du graphe (Panel-Level, Niveau 1) :
    - 9 nœuds par essai : 8 panels contexte + 1 candidat testé
    - Arêtes de rangée : P0↔P1, P1↔P2, P3↔P4, P4↔P5, P6↔P7, P7↔Candidat
    - Arêtes de colonne : P0↔P3, P3↔P6, P1↔P4, P4↔P7, P2↔P5, P5↔Candidat
    - Total : 12 arêtes (bidirectionnelles → 24 arêtes dirigées)
    
Processus de scoring :
    Pour chaque candidat Ci (i=0..7) :
        1. Construire le graphe avec Ci à la position "?"
        2. Message passing avec GAT (attention entre voisins)
        3. Calculer un score de cohérence pour ce graphe
    → 8 scores → softmax → prédiction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch


class RPMGraphBuilder:
    """
    Construit le graphe RPM pour la matrice 3×3.
    
    Disposition de la matrice :
        P0  P1  P2      (Rangée 0)
        P3  P4  P5      (Rangée 1)
        P6  P7  [?]     (Rangée 2, position 8 = candidat)
    """
    
    # Arêtes de la matrice 3×3 (indices 0-8, où 8 = position "?")
    # Format : (source, destination)
    ROW_EDGES = [
        (0, 1), (1, 2),     # Rangée 0
        (3, 4), (4, 5),     # Rangée 1
        (6, 7), (7, 8),     # Rangée 2 (8 = candidat)
    ]
    
    COL_EDGES = [
        (0, 3), (3, 6),     # Colonne 0
        (1, 4), (4, 7),     # Colonne 1
        (2, 5), (5, 8),     # Colonne 2 (8 = candidat)
    ]
    
    @classmethod
    def get_edge_index(cls):
        """
        Retourne l'edge_index pour le graphe RPM.
        
        Les arêtes sont bidirectionnelles (non-dirigées) :
        si P0→P1 existe, alors P1→P0 aussi.
        
        Returns:
            Tensor de shape (2, num_edges) — format COO de PyG
        """
        edges = cls.ROW_EDGES + cls.COL_EDGES
        
        # Rendre bidirectionnel : ajouter les arêtes inverses
        all_edges = []
        for src, dst in edges:
            all_edges.append((src, dst))
            all_edges.append((dst, src))
        
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        return edge_index  # shape: (2, 24)
    
    @classmethod
    def build_graph(cls, context_embeddings, candidate_embedding):
        """
        Construit un graphe PyG pour un candidat donné.
        
        Args:
            context_embeddings: Tensor (8, embed_dim) — les 8 panels contexte
            candidate_embedding: Tensor (embed_dim,) — un candidat
        
        Returns:
            torch_geometric.data.Data — le graphe avec 9 nœuds
        """
        # Assembler les 9 nœuds : 8 contexte + 1 candidat
        node_features = torch.cat([
            context_embeddings,                    # (8, embed_dim)
            candidate_embedding.unsqueeze(0),      # (1, embed_dim)
        ], dim=0)  # → (9, embed_dim)
        
        edge_index = cls.get_edge_index().to(node_features.device)
        
        return Data(x=node_features, edge_index=edge_index)
    
    @classmethod
    def build_batch(cls, context_embeddings, candidate_embeddings):
        """
        Construit un batch de 8 graphes (un par candidat) pour un problème RPM.
        
        Args:
            context_embeddings: Tensor (8, embed_dim) — les 8 panels contexte
            candidate_embeddings: Tensor (8, embed_dim) — les 8 candidats
        
        Returns:
            torch_geometric.data.Batch — batch de 8 graphes
        """
        graphs = []
        for i in range(8):
            graph = cls.build_graph(context_embeddings, candidate_embeddings[i])
            graphs.append(graph)
        
        return Batch.from_data_list(graphs)


class GNNReasoner(nn.Module):
    """
    GNN qui raisonne sur le graphe RPM pour scorer les candidats.
    
    Architecture :
        1. Couches GAT (Graph Attention) : message passing avec attention
        2. Readout : agrégation globale du graphe en un seul vecteur
        3. Scorer : MLP qui calcule un score de cohérence
    
    Args:
        embed_dim (int): Dimension des embeddings (sortie du CNN)
        hidden_dim (int): Dimension cachée du GAT
        num_heads (int): Nombre de têtes d'attention dans le GAT
        num_layers (int): Nombre de couches GAT
        dropout (float): Taux de dropout
    
    Exemple:
        >>> gnn = GNNReasoner(embed_dim=256, hidden_dim=128)
        >>> context = torch.randn(8, 256)     # 8 panels contexte
        >>> candidates = torch.randn(8, 256)  # 8 candidats
        >>> scores = gnn(context, candidates)  # (8,) — un score par candidat
    """
    
    def __init__(self, embed_dim=256, hidden_dim=128, num_heads=4,
                 num_layers=2, dropout=0.1):
        super(GNNReasoner, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # =====================================================================
        # Couches GAT (Graph Attention Network)
        # =====================================================================
        # Le GAT apprend des coefficients d'attention entre voisins,
        # permettant au modèle de pondérer l'importance relative
        # de chaque relation (rangée vs colonne).
        
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Première couche : embed_dim → hidden_dim × num_heads
        self.gat_layers.append(
            GATConv(embed_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True)
        )
        self.norms.append(nn.LayerNorm(hidden_dim * num_heads))
        
        # Couches suivantes : hidden_dim × num_heads → hidden_dim × num_heads
        for _ in range(num_layers - 1):
            self.gat_layers.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads,
                        dropout=dropout, concat=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim * num_heads))
        
        # =====================================================================
        # Scorer : vecteur global du graphe → score scalaire
        # =====================================================================
        # global_mean_pool agrège les 9 nœuds en 1 seul vecteur,
        # puis le MLP projette ce vecteur en un score de cohérence.
        
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # → 1 score par graphe
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, context_embeddings, candidate_embeddings):
        """
        Score chaque candidat en le plaçant dans le graphe RPM.
        
        Args:
            context_embeddings: Tensor (batch, 8, embed_dim) ou (8, embed_dim)
            candidate_embeddings: Tensor (batch, 8, embed_dim) ou (8, embed_dim)
        
        Returns:
            Tensor (batch, 8) ou (8,) — scores pour chaque candidat
        """
        has_batch = context_embeddings.dim() == 3
        
        if not has_batch:
            context_embeddings = context_embeddings.unsqueeze(0)      # (1, 8, D)
            candidate_embeddings = candidate_embeddings.unsqueeze(0)  # (1, 8, D)
        
        batch_size = context_embeddings.size(0)
        all_scores = []
        
        for b in range(batch_size):
            # Construire un batch de 8 graphes (un par candidat)
            graph_batch = RPMGraphBuilder.build_batch(
                context_embeddings[b],      # (8, embed_dim)
                candidate_embeddings[b],    # (8, embed_dim)
            )
            graph_batch = graph_batch.to(context_embeddings.device)
            
            # --- Message Passing avec GAT ---
            x = graph_batch.x                    # (9*8=72, embed_dim)
            edge_index = graph_batch.edge_index  # (2, 24*8=192)
            batch_idx = graph_batch.batch         # (72,) — quel nœud ∈ quel graphe
            
            for gat, norm in zip(self.gat_layers, self.norms):
                x = gat(x, edge_index)           # GAT message passing
                x = norm(x)                       # LayerNorm
                x = F.elu(x)                      # Activation
                x = self.dropout(x)
            
            # --- Readout : agrégation globale ---
            # Moyenne des 9 nœuds de chaque graphe → 1 vecteur par graphe
            graph_emb = global_mean_pool(x, batch_idx)  # (8, hidden_dim*heads)
            
            # --- Scoring ---
            scores = self.scorer(graph_emb).squeeze(-1)  # (8,)
            all_scores.append(scores)
        
        all_scores = torch.stack(all_scores)  # (batch, 8)
        
        if not has_batch:
            all_scores = all_scores.squeeze(0)  # (8,)
        
        return all_scores


# =============================================================================
# Script de test
# =============================================================================
if __name__ == "__main__":
    import os
    import sys
    
    print("=" * 60)
    print("Test du GNN Raisonneur")
    print("=" * 60)
    
    # --- Test 1 : Construction du graphe ---
    print("\n--- Test 1 : Construction du graphe RPM ---")
    edge_index = RPMGraphBuilder.get_edge_index()
    print(f"  edge_index shape : {edge_index.shape}")  # (2, 24)
    print(f"  Nombre d'arêtes  : {edge_index.shape[1]} (12 bidirectionnelles)")
    print(f"  Nœuds connectés  : {edge_index.min().item()} à {edge_index.max().item()}")
    
    # --- Test 2 : Build un graphe pour un candidat ---
    print("\n--- Test 2 : Construire un graphe (1 candidat) ---")
    context = torch.randn(8, 256)
    candidate = torch.randn(256)
    graph = RPMGraphBuilder.build_graph(context, candidate)
    print(f"  Nœuds     : {graph.x.shape}")            # (9, 256)
    print(f"  Arêtes    : {graph.edge_index.shape}")    # (2, 24)
    
    # --- Test 3 : Build batch de 8 graphes ---
    print("\n--- Test 3 : Batch de 8 graphes (8 candidats) ---")
    candidates = torch.randn(8, 256)
    batch_graph = RPMGraphBuilder.build_batch(context, candidates)
    print(f"  Nœuds total     : {batch_graph.x.shape}")          # (72, 256)
    print(f"  Arêtes total    : {batch_graph.edge_index.shape}") # (2, 192)
    print(f"  Batch index     : {batch_graph.batch.shape}")      # (72,)
    print(f"  Num graphes     : {batch_graph.num_graphs}")       # 8
    
    # --- Test 4 : Forward pass du GNN ---
    print("\n--- Test 4 : Forward pass (sans batch) ---")
    gnn = GNNReasoner(embed_dim=256, hidden_dim=128, num_heads=4, num_layers=2)
    
    total_params = sum(p.numel() for p in gnn.parameters())
    print(f"  Paramètres GNN  : {total_params:,}")
    
    with torch.no_grad():
        scores = gnn(context, candidates)
    print(f"  Context shape   : {context.shape}")      # (8, 256)
    print(f"  Candidates shape: {candidates.shape}")   # (8, 256)
    print(f"  Scores shape    : {scores.shape}")       # (8,)
    print(f"  Scores          : {scores}")
    print(f"  Prédiction      : candidat {scores.argmax().item()}")
    
    # --- Test 5 : Forward pass avec batch ---
    print("\n--- Test 5 : Forward pass (batch de 4) ---")
    context_batch = torch.randn(4, 8, 256)
    candidates_batch = torch.randn(4, 8, 256)
    with torch.no_grad():
        scores_batch = gnn(context_batch, candidates_batch)
    print(f"  Scores shape    : {scores_batch.shape}")   # (4, 8)
    print(f"  Prédictions     : {scores_batch.argmax(dim=1)}")
    
    # --- Test 6 : Pipeline complet Encodeur → GNN ---
    print("\n--- Test 6 : Pipeline Encodeur → GNN ---")
    try:
        from models.encoder_decoder import RavenEncoderDecoder
        from data.raven_dataset import RavenDataset
        
        DATA_DIR = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "data", "raw"
        )
        
        # Charger un problème (16 panels individuels)
        dataset = RavenDataset(DATA_DIR, split="train", config="center_single", limit_per_config=1)
        
        # Empiler les 16 images du problème en un batch
        images_list = []
        for i in range(16):
            img, _, _ = dataset[i]  # (1, 160, 160), target, p_type
            images_list.append(img)
        images_batch = torch.stack(images_list)  # (16, 1, 160, 160)
        
        # Encodeur : images → vecteurs structurés (16, 10, 13)
        encoder = RavenEncoderDecoder(num_slots=10, feat_dim=13)
        with torch.no_grad():
            z, _ = encoder(images_batch)  # z: (16, 10, 13)
        
        # Aplatir les vecteurs structurés pour le GNN : (16, 130)
        embed_dim = 10 * 13  # = 130
        embeddings = z.view(16, embed_dim)
        
        # Séparer contexte et candidats
        context_emb = embeddings[:8]    # (8, 130) — panels contexte
        candidate_emb = embeddings[8:]  # (8, 130) — panels candidats
        
        # GNN : embeddings → scores (adapter embed_dim = 130)
        gnn_pipeline = GNNReasoner(embed_dim=embed_dim, hidden_dim=128, num_heads=4, num_layers=2)
        with torch.no_grad():
            scores = gnn_pipeline(context_emb, candidate_emb)  # (8,)
        
        print(f"  Images          : {images_batch.shape}")
        print(f"  Encodeur z      : {z.shape}")
        print(f"  Embeddings      : {embeddings.shape}")
        print(f"  Context         : {context_emb.shape}")
        print(f"  Candidates      : {candidate_emb.shape}")
        print(f"  Scores          : {scores}")
        print(f"  Prédiction      : candidat {scores.argmax().item()}")
        print("  ✓ Pipeline RavenDataset → Encodeur → GNN fonctionne !")
        
    except Exception as e:
        import traceback
        print(f"  ERREUR : {e}")
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Tests terminés !")
