import torch.nn as nn
from torch.nn.functional import normalize
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter



class Encoder(nn.Module):
    def __init__(self, input_dim, recon_fea_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, recon_fea_dim),
            nn.BatchNorm1d(recon_fea_dim),
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, recon_fea_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(recon_fea_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

class Network(nn.Module):
    def __init__(self, view_num, view_dims, recon_fea_dim, z_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view_num):
            self.encoders.append(Encoder(view_dims[v], recon_fea_dim).to(device))
            self.decoders.append(Decoder(view_dims[v], recon_fea_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(recon_fea_dim, z_dim),
        )
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(recon_fea_dim, class_num),
            nn.Softmax(dim=1)
        )
        self.view_num = view_num
        self.view_dims = view_dims

    def forward_singleh(self, xs):  # Forward pass for single-view features
        z_spec = []
        h_spec = []
        for v in range(self.view_num):
            x = xs[v]
            h = self.encoders[v](x)
            z = normalize(self.feature_contrastive_module(h), dim=1)
            z_spec.append(z)
            h_spec.append(h)
        # Get reconstructed and contrastive representations for each view
        return h_spec, z_spec

    def forward_commonZ(self, xs, miss_vecs):
        z_spec = []
        h_spec = []
        h_fea, z_fea = self.forward_singleh(xs)
        for v in range(self.view_num):
            h_spec.append((torch.mul((h_fea[v]).t(), miss_vecs[v]).t()).float())
            z_spec.append((torch.mul((z_fea[v]).t(), miss_vecs[v]).t()).float())
        # Calculate common z
        h_com = (torch.mul(sum(h_spec).t(), (1 / sum(miss_vecs))).t()).float()
        z_com = (torch.mul(sum(z_spec).t(), (1 / sum(miss_vecs))).t()).float()
        return h_com, z_com

    def forward_recon(self, xs, miss_vecs):
        # Representations, reconstructed samples, and original samples for complete instances in each view
        z_spec, z_all_spec, xr_spec, xs_com = [], [], [], []

        for v in range(self.view_num):
            cur_com_idx = torch.where(miss_vecs[v] > 0)[0]
            z = self.encoders[v](xs[v])
            xr = self.decoders[v](z[cur_com_idx])
            z_spec.append(z[cur_com_idx]) # Latent representations for non-missing samples
            z_all_spec.append(z)  # Latent representations for all samples (including missing)
            xr_spec.append(xr) # Reconstructed samples for non-missing samples
            xs_com.append(xs[v][cur_com_idx]) # Original non-missing samples

        return z_spec, z_all_spec, xr_spec, xs_com


class Construct_Graph(nn.Module):
    def __init__(self, gamma, k_neighber):
        super(Construct_Graph, self).__init__()
        self.gamma = gamma
        self.k_neighber = k_neighber

    def compute_similarity(self, x):
        x_i = x.unsqueeze(1)  # shape: (n_samples, 1, n_features)
        x_j = x.unsqueeze(0)  # shape: (1, n_samples, n_features)
        dist_squared = torch.sum((x_i - x_j) ** 2, dim=2)  # shape: (n_samples, n_samples)
        S = torch.exp(- dist_squared / self.gamma)
        return S

    # Construct and normalize the adjacency matrix of a K-nearest neighbor graph
    def normalize(self, mx):
        rowsum = torch.sum(mx, dim=1)
        rowsum_inv = torch.pow(rowsum, -1)
        rowsum_inv[rowsum_inv == float('inf')] = 0.  # Handle division by zero
        D_inv_sqrt = torch.diag_embed(rowsum_inv)
        mx_normalized = torch.matmul(D_inv_sqrt, mx)
        return mx_normalized

    def construct_knn_graph_spec(self, x):
        """
        Args:
            x (torch.Tensor): Input feature matrix of shape (N, F), where N is the number of samples and F is the feature dimension.
        Returns:
            adjacency_matrix (torch.Tensor): Adjacency matrix of the K-nearest neighbor graph.
            A_hat_sparse (torch.sparse_coo_tensor): Normalized adjacency matrix of the K-nearest neighbor graph (sparse representation).
        """
        similarity_matrix = self.compute_similarity(x)
        # Get number of nodes
        num_nodes = similarity_matrix.size(0)
        k = self.k_neighber

        # Set diagonal to negative infinity to exclude self-loops
        diag_mask = torch.eye(num_nodes, dtype=torch.bool, device=similarity_matrix.device)
        similarity_matrix.masked_fill_(diag_mask, float('-inf'))

        # Find indices of the top k maximum values for each node
        topk_values, topk_indices = torch.topk(similarity_matrix, k=min(k, num_nodes - 1), dim=1)

        # Initialize adjacency matrix
        adjacency_matrix = torch.zeros_like(similarity_matrix)

        # Use advanced indexing to mark connections
        row_indices = torch.arange(num_nodes, device=similarity_matrix.device).unsqueeze(1).expand(-1,
                                                                                                   topk_indices.size(
                                                                                                       1))
        # Ensure shapes of row_indices and topk_indices match
        assert row_indices.shape == topk_indices.shape, "Shape mismatch between row_indices and topk_indices"

        adjacency_matrix[row_indices, topk_indices] = 1
        adjacency_matrix[topk_indices, row_indices] = 1  # Ensure the adjacency matrix is symmetric

        A_hat = self.normalize(adjacency_matrix)
        # Convert A_hat to a sparse tensor
        A_hat_sparse = torch.sparse_coo_tensor(A_hat.nonzero().t(), A_hat[A_hat.nonzero(as_tuple=True)],
                                               size=A_hat.size())
        return adjacency_matrix, A_hat_sparse

class GraphConvolution(nn.Module):
    def __init__(self, in_features_dim, out_features_dim, activation='selu', skip_connection=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features_dim
        self.out_features = out_features_dim
        self.skip_connection = skip_connection
        self.activation = nn.SELU() if activation == 'selu' else nn.Identity()
        self.kernel = nn.Parameter(torch.Tensor(in_features_dim, out_features_dim))
        self.bias = nn.Parameter(torch.Tensor(out_features_dim))
        if self.skip_connection:
            self.skip_weight = nn.Parameter(torch.Tensor(out_features_dim))
        else:
            self.register_parameter('skip_weight', None)

        # Initialize parameters
        nn.init.xavier_uniform_(self.kernel)
        nn.init.zeros_(self.bias)
        if self.skip_connection:
            nn.init.ones_(self.skip_weight)

    def forward(self, features, norm_adjacency):
        output = torch.matmul(features, self.kernel)
        if self.skip_connection:
            output = output * self.skip_weight + torch.sparse.mm(norm_adjacency, output)
        else:
            output = torch.sparse.mm(norm_adjacency, output)
        output = output + self.bias
        return self.activation(output)

class GCN(nn.Module):
    def __init__(self, in_features_dim, hidden_features_dim, out_features_dim, activation='selu'):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(in_features_dim, hidden_features_dim, activation)
        self.gc2 = GraphConvolution(hidden_features_dim, out_features_dim,
                                    activation='identity')  # No activation on last layer

    def forward_spec(self, feature, norm_adjacency):
        x = self.gc1.forward(feature, norm_adjacency)
        x = F.relu(x)  # Apply ReLU activation
        embedding = self.gc2.forward(x, norm_adjacency)
        return embedding


class MGC(nn.Module):
    def __init__(self, n_clusters, in_features, device, collapse_regularization, dropout_rate=0.0,
                 do_unpooling=False):
        super(MGC, self).__init__()
        self.n_clusters = n_clusters
        self.collapse_regularization = collapse_regularization
        self.dropout_rate = dropout_rate
        self.do_unpooling = do_unpooling
        self.transform = nn.Sequential(
            nn.Linear(in_features, n_clusters, bias=True),
            nn.Dropout(dropout_rate)
        ).to(device)
        # Initialize layers with appropriate initializers
        nn.init.orthogonal_(self.transform[0].weight)  # Orthogonal initialization
        nn.init.zeros_(self.transform[0].bias)  # Zero initialization for bias


        self.activation = nn.SELU()

    def compute_loss(self, features, adjacency):  # x_embedding, unnormalize_adjacency
        assignments = torch.softmax(self.transform(features), dim=1)


        valid_assignments = assignments  # b * k
        valid_adjacency = adjacency.to_sparse()


        degrees = torch.sum(adjacency, dim=0)  # Sum over each column
        degrees = degrees.reshape(-1, 1)

  
        number_of_edges = degrees.sum()

        graph_pooled = torch.transpose(torch.sparse.mm(valid_adjacency, valid_assignments), 0, 1)
        graph_pooled = torch.matmul(graph_pooled, valid_assignments)

        normalizer_left = torch.matmul(valid_assignments.transpose(0, 1), degrees)
        normalizer_right = torch.matmul(degrees.transpose(0, 1), valid_assignments)
        normalizer = torch.matmul(normalizer_left, normalizer_right) / 2 / number_of_edges
        spectral_loss = - torch.trace(graph_pooled - normalizer) / 2 / number_of_edges

        return assignments, (spectral_loss)

    def Clustering(self, features):
        assignments = torch.softmax(self.transform(features), dim=1)
        return assignments



class FreeCSL(nn.Module):
    def __init__(self, args):
        super(FreeCSL, self).__init__()
        self.args = args
        self.AE = Network(args.view_num, args.view_dims, args.recon_fea_dim, args.z_dim, args.cluster_num, args.device)
        self.clu_H_layer = Parameter(torch.randn(args.cluster_num, args.recon_fea_dim).cuda(), requires_grad=True)
        self.clu_Z_layer = Parameter(torch.randn(args.cluster_num, args.recon_fea_dim).cuda(), requires_grad=True)
        self.criterionMSE = torch.nn.MSELoss()

        self.KNN_Graph = Construct_Graph(args.gamma, args.K_neighber)
        self.GCN_list = nn.ModuleList()
        for v in range(args.view_num):
            gcn_instance = GCN(args.recon_fea_dim, args.graph_h_dim, args.graph_out_dim)  # Create a new GCN instance
            self.GCN_list.append(gcn_instance)
        self.MGC = MGC(args.cluster_num, args.graph_out_dim, args.device, args.collapse_regularization)

        self.linear_ctoe = nn.Linear(args.recon_fea_dim, args.graph_out_dim)

    def get_Single_reconHs(self, xs): # Get reconstructed feature vectors for V views
        Hs,_ = self.AE.forward_singleh(xs)
        return Hs

    def get_Single_constrZs(self, xs):
        _, Zs = self.AE.forward_singleh(xs)
        return Zs

    @torch.no_grad()
    def distributed_sinkhorn(self, score):
        Q = torch.exp(score / self.args.epsilon).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.args.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K
            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    def Constr_loss(self, Z_m, Z_n, C, tau, num_samples):
        # prototype scores: 2BxK
        Z = torch.cat((Z_m, Z_n),dim = 0)
        Z_scores = torch.matmul(Z, C.t())
        Z_scores_m = Z_scores[:num_samples]
        Z_scores_n = Z_scores[num_samples:]
        # compute assignments (pseudo-labels)
        with torch.no_grad():
            q_m = self.distributed_sinkhorn(Z_scores_m)
            q_n = self.distributed_sinkhorn(Z_scores_n)
        # convert scores to probabilities
        p_m = torch.softmax((Z_scores_m / tau), dim=1)
        p_n = torch.softmax((Z_scores_n / tau), dim=1)
        # swap prediction problem
        constr_loss = - 0.5 * torch.mean(q_m * torch.log(p_n) + q_n * torch.log(p_m))
        return constr_loss

    def q_assignments(self, h_valid):
        # h_valid = self.get_Single_reconHs(xs)
        q_tmp = 1.0 / (1.0 + torch.sum(torch.pow(h_valid.unsqueeze(1) - self.clu_Z_layer, 2), 2) / self.args.alpha)
        q_tmp = q_tmp.pow((self.args.alpha + 1.0) / 2.0)
        q = (q_tmp.t() / torch.sum(q_tmp, 1)).t()

        return q

    def compute_q(self, h_valid):
        """
        Computes the soft assignment matrix q_ij.
        :param z: Sample feature matrix, shape (N, D).
        :param u: Cluster center matrix, shape (M, D).
        :param v: Parameter v, controls the shape of the distribution.
        :return: Soft assignment matrix q, shape (N, M).
        """
        # Calculate squared Euclidean distance between samples and cluster centers
        u = self.clu_Z_layer
        z = h_valid
        dist = torch.cdist(z, u, p=2) ** 2  # (N, M)
        # Calculate the numerator
        numerator = (1 + dist / self.args.alpha).pow(-(self.args.alpha + 1) / 2)
        # Calculate the denominator
        denominator = torch.sum(numerator, dim=1, keepdim=True)
        # Calculate the soft assignment matrix q_ij
        q = numerator / denominator

        return q
    ###### Reconstruction Loss ######
    def train_Recon(self, xs, miss_vecs):
        z_spec, _, xr_spec, xs_com = self.AE.forward_recon(xs, miss_vecs)
        loss_list_recon = [self.criterionMSE(xr_spec[v], xs_com[v]) for v in range(self.args.view_num)]
        loss_recon = sum(loss_list_recon)
        return loss_recon

    def train_Constr(self, xs, miss_vecs):
        Z_spec_all_real, z_spec_all = self.AE.forward_singleh(xs) # z_spec_all is actually h from the contrastive layer
        # batch_samples_num = z_spec_all[0].shape[0]
        pairs_views = [(m, n) for m in range(self.args.view_num) for n in range(m, self.args.view_num) if m != n]
        Loss_views_constra = 0.0
        for m, n in pairs_views:
            # Step 1: Find paired samples from the two views
            mask_mn = (miss_vecs[m] > 0) & (miss_vecs[n] > 0)  # Samples present in both view m and n

            # Directly extract paired samples using the mask
            Z_m = z_spec_all[m][mask_mn]
            Z_n = z_spec_all[n][mask_mn]

            pairs_samples_idx = Z_m.shape[0]  # Number of paired samples
            # Step 2: Perform prototype contrastive learning via mutual prediction on paired samples
            # For view i, compute probability label p; for view j, use it as pseudo-label q.
            loss_two_Constr = self.Constr_loss(Z_m, Z_n, self.clu_H_layer, self.args.tau, pairs_samples_idx)
            Loss_views_constra += loss_two_Constr
        return z_spec_all, Loss_views_constra


    def train_Graph(self, xs, miss_vecs):
        h_spec = self.get_Single_reconHs(xs)
        z_spec = self.get_Single_constrZs(xs)
        assign_result = []
        MGC_loss = 0.0
        for v in range(self.args.view_num):
            valid_indices = torch.nonzero(miss_vecs[v].squeeze() != 0, as_tuple=False).squeeze(dim=1)
            z_valid = z_spec[v][valid_indices]
            x_valid = xs[v][valid_indices]
            h_valid = h_spec[v][valid_indices]
            A, A_sparse = self.KNN_Graph.construct_knn_graph_spec(x_valid)

            x_embedding = self.GCN_list[v].forward_spec(h_valid, A_sparse)

            q_assignment = self.compute_q(z_valid)
            assign_x, loss_spec = self.MGC.compute_loss(x_embedding, A)# Using unnormalized_A
            epsilon = 1e-10
            assign_x = assign_x + epsilon
            q_assignment = q_assignment + epsilon
            kl_loss = F.kl_div(assign_x.log(), q_assignment, reduction='batchmean')
            assignments = torch.zeros(len(xs[v]), self.args.cluster_num, dtype=torch.float32).cuda()
            assignments[valid_indices] = assign_x
            assign_result.append(assignments)

            MGC_loss += loss_spec + self.args.collapse_regularization * kl_loss

        return assign_result, MGC_loss



    def Clustering(self, xs, miss_vecs):
        h_spec = self.get_Single_reconHs(xs)
        z_spec = self.get_Single_constrZs(xs)
        assign_result = []
        for v in range(self.args.view_num):
            valid_indices = torch.nonzero(miss_vecs[v].squeeze() != 0, as_tuple=False).squeeze(dim=1)
            x_valid = xs[v][valid_indices]
            h_valid = h_spec[v][valid_indices]
            z_valid = z_spec[v][valid_indices]
            _, norm_A_sparse = self.KNN_Graph.construct_knn_graph_spec(x_valid)
            x_embedding = self.GCN_list[v].forward_spec(h_valid, norm_A_sparse)
            assign_x = self.MGC.Clustering(x_embedding)
            assignments = torch.zeros(len(xs[v]), self.args.cluster_num, dtype=torch.float32).cuda()
            assignments[valid_indices] = assign_x
            assign_result.append(assignments)
        return assign_result






