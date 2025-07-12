# === Standard Library ===
import os
import pickle
from collections import defaultdict

# === Third-party Libraries ===
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from tqdm import tqdm
from einops import rearrange

# === Project Modules ===
from .biomedclip import BiomedCLIP
from .layers import AttentionApproximation


class PSA(nn.Module):
    """
    Prototypical Semantic Aggregation (PSA) module for multi-modal representation learning.
    Integrates vision-language features via clustering and attention-weighted aggregation.
    """

    def __init__(self, args):
        super(PSA, self).__init__()
        self.args = args
        self.focus = "position"
        self.agg = args.agg
        self.pretrain = "contrastive"

        self.matrix = torch.randn(args.num_classes, args.num_prototypes, args.prototype_dim)
        self.memory_text = torch.randn(args.num_classes, args.num_prototypes, 256, 768)

        self.encoder = BiomedCLIP(args).to(args.device)
        self.cluster = KMeans(n_clusters=args.num_prototypes)
        self.approx = AttentionApproximation(args.text_dim, args.num_candidate, args.project_dim).to(args.device)

    def fit(self, data_loader):
        """
        Fit PSA by clustering feature vectors and constructing prototypes.

        Args:
            data_loader (DataLoader): Iterable over training data with image, text, and pseudolabels.
        """
        if self.load():
            return

        if self.pretrain:
            self.encoder.contrastive_learning(data_loader)

        self.encoder.eval()
        image_space, text_space, label_space = defaultdict(list), defaultdict(list), defaultdict(list)

        for batch in tqdm(data_loader, desc="Prototypical Learning", leave=False):
            image, text, pseudolabel = batch["image"], batch["text"], batch["pseudolabel"]
            pseudolabel = pseudolabel[self.focus]

            image_emb = self.encoder.encode_image(image).detach()
            text_emb = self.encoder.encode_text(text).detach()
            image_feat = self.encoder.encode_image_feature(image).detach()
            text_feat = self.encoder.encode_text_feature(text).detach()

            for i in range(pseudolabel.shape[0]):
                for j in range(pseudolabel.shape[1]):
                    if pseudolabel[i, j] == 1:
                        image_space[j].append(image_feat[i])
                        text_space[j].append(text_feat[i])
                        label_space[j].append(torch.cat((image_emb[i], text_emb[i]), dim=0))

        for label, reps in tqdm(label_space.items(), desc="Prototype Constructing", leave=False):
            flattened = torch.stack([rep.flatten() for rep in reps]).cpu().numpy()
            clusters = self.cluster.fit_predict(flattened)

            label_protos, text_protos = defaultdict(list), defaultdict(list)
            for idx, cluster_id in enumerate(clusters):
                label_protos[cluster_id].append(label_space[label][idx])
                text_protos[cluster_id].append(text_space[label][idx])

            for cid in label_protos:
                self.matrix[label, cid] = torch.stack(label_protos[cid], dim=0).mean(dim=0).float()
                self.memory_text[label, cid] = torch.stack(text_protos[cid], dim=0).mean(dim=0).float()

        self.save()
        self.load()

    def save(self):
        """
        Serialize prototype matrix and encoder weights to disk.
        """
        os.makedirs(self.args.prototype_save_path, exist_ok=True)
        filename_proto = f"{self.args.prototype_save_path}/{self.args.prototype_save_filename}_{self.args.dataset_name.lower()}_{self.args.num_classes}_{self.args.num_prototypes}_{self.args.prototype_dim}_{self.pretrain}.pkl"
        filename_clip = filename_proto.replace('.pkl', '.pth')

        state = {
            "matrix": self.matrix.detach(),
            "memory_text": self.memory_text.detach()
        }

        with open(filename_proto, 'wb') as f:
            pickle.dump(state, f)
        torch.save(self.encoder.state_dict(), filename_clip)

        print(f"[✓] Prototypes saved to {filename_proto}")
        print(f"[✓] Encoder weights saved to {filename_clip}")

    def load(self, filename_proto=None, filename_clip=None):
        """
        Load precomputed prototypes and encoder weights from disk.

        Returns:
            bool: True if both files loaded successfully, else False.
        """
        if filename_proto is None or filename_clip is None:
            filename_proto = f"{self.args.prototype_save_path}/{self.args.prototype_save_filename}_{self.args.dataset_name.lower()}_{self.args.num_classes}_{self.args.num_prototypes}_{self.args.prototype_dim}_{self.pretrain}.pkl"
            filename_clip = filename_proto.replace('.pkl', '.pth')

        if os.path.exists(filename_proto) and os.path.exists(filename_clip):
            print(f"[✓] Loading prototypes from {filename_proto}")
            with open(filename_proto, 'rb') as f:
                state = pickle.load(f)

            self.matrix = nn.Parameter(state["matrix"].to(self.args.device))
            self.memory_text = state["memory_text"].to(self.args.device)

            part1 = nn.Parameter(self.matrix[:, :, :self.args.clip_dim], requires_grad=False)
            part2 = nn.Parameter(self.matrix[:, :, self.args.clip_dim:], requires_grad=True)
            self.matrix = nn.Parameter(torch.cat([part1, part2], dim=-1))

            return True
        else:
            print(f"[✗] Missing prototype file: {filename_proto} or {filename_clip}")
            return False

    def similarity(self, matrix_slice, img_feat, similarity_type='cosine'):
        """
        Compute similarity between image features and prototype matrix.
        """
        if similarity_type == 'cosine':
            matrix_norm = matrix_slice / (matrix_slice.norm(dim=1, keepdim=True) + 1e-8)
            img_norm = img_feat / (img_feat.norm(dim=0, keepdim=True) + 1e-8)
            return torch.einsum('ij,j->i', matrix_norm, img_norm)
        elif similarity_type == 'euclidean':
            return -torch.norm(matrix_slice - img_feat, dim=1)
        else:
            raise ValueError(f"Invalid similarity type: {similarity_type}")

    def top_k_candidates(self, image, similarity_type='cosine'):
        """
        Retrieve top-k prototype candidates based on similarity.

        Args:
            image (Tensor): Image embeddings (B, D).
            similarity_type (str): 'cosine' or 'euclidean'.

        Returns:
            Tuple[List[Tuple[int, int]], List[float]]: Candidate indices and scores.
        """
        k = self.args.num_candidate
        candidates_indices, candidates_scores = [], []
        matrix_slice = rearrange(self.matrix[:, :, :self.args.clip_dim], 'L H D -> (L H) D')

        for img_feat in image:
            scores = self.similarity(matrix_slice.to(img_feat.device), img_feat, similarity_type)
            topk_scores, topk_flat = torch.topk(scores, k=k)
            indices = [(i // self.args.num_prototypes, i % self.args.num_prototypes) for i in topk_flat]
            candidates_indices.append(indices)
            candidates_scores.append(topk_scores)
        return candidates_indices, candidates_scores

    def respond(self, candidates_indices, candidates_scores, mode='sum'):
        """
        Aggregate memory_text vectors from top-k prototypes.

        Args:
            mode (str): Aggregation method: 'sum', 'linear', or 'attention'.

        Returns:
            Tensor: Aggregated representation per image.
        """
        B = len(candidates_indices)
        if mode == 'sum':
            agg_feat = torch.zeros(B, self.args.bert_length, self.args.feature_dim)
            for i, (indices, scores) in enumerate(zip(candidates_indices, candidates_scores)):
                vectors = torch.stack([self.memory_text[r, c] for r, c in indices])
                weights = torch.softmax(scores, dim=0).view(-1, 1, 1)
                agg_feat[i] = (vectors * weights).sum(dim=0)
            return agg_feat

        elif mode in {'linear', 'attention'}:
            max_k = max(len(x) for x in candidates_indices)
            proto_feat = torch.zeros(B, max_k, self.args.prototype_dim // 2)
            for i, indices in enumerate(candidates_indices):
                vectors = torch.stack([self.matrix[r, c, self.args.clip_dim:] for r, c in indices])
                proto_feat[i, :vectors.size(0)] = vectors
            return proto_feat

        else:
            raise NotImplementedError(f"Aggregation mode '{mode}' not implemented.")

    def query(self, images, image_emb=None, image_feature=None):
        """
        Query the PSA with images and return aggregated features.

        Args:
            images (Tensor): Input images.
            image_emb (Tensor): Optional precomputed embeddings.
            image_feature (Tensor): Optional feature maps for attention.

        Returns:
            Tensor: Aggregated semantic response.
        """
        image_emb = self.encoder.encode_image(images).detach() if image_emb is None else image_emb.detach()
        indices, scores = self.top_k_candidates(image_emb)
        agg_feat = self.respond(indices, scores, self.agg)
        if self.agg == "attention" and image_feature is not None:
            return self.approx(agg_feat.to(image_feature.device), image_feature)
        return agg_feat.to(images.device)

    def visualize(self, reduction_method="t-SNE", n_components=2, save_path="vector_space_visualization.png"):
        """
        Visualize the learned prototype matrix in 2D or 3D space.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from mpl_toolkits.mplot3d import Axes3D

        mat = self.matrix.detach().cpu().numpy().reshape(-1, self.matrix.shape[-1])
        labels = np.repeat(np.arange(self.args.num_classes), self.args.num_prototypes)

        reducer = TSNE(n_components=n_components, random_state=42) if reduction_method == "t-SNE" else PCA(n_components=n_components)
        reduced = reducer.fit_transform(mat)

        plt.figure(figsize=(12, 8))
        if n_components == 2:
            for i in range(self.args.num_classes):
                idx = labels == i
                plt.scatter(reduced[idx, 0], reduced[idx, 1], label=f"Class {i}", alpha=0.7)
            plt.xlabel("Dim 1"), plt.ylabel("Dim 2"), plt.legend()
        elif n_components == 3:
            ax = plt.figure().add_subplot(111, projection='3d')
            for i in range(self.args.num_classes):
                idx = labels == i
                ax.scatter(reduced[idx, 0], reduced[idx, 1], reduced[idx, 2], label=f"Class {i}", alpha=0.7)
            ax.set_xlabel("Dim 1"), ax.set_ylabel("Dim 2"), ax.set_zlabel("Dim 3"), ax.legend()

        plt.title(f"Prototype Space ({reduction_method})")
        plt.grid(True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path
