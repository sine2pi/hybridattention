
class SparseAttention(nn.Module):
    def __init__(self, base, dims, head, max_dist, sparsity_factor):
        super().__init__()
        self.head = head
        self.multihead_attn = nn.MultiheadAttention(dims, head)
        self.sparsity_factor = sparsity_factor

    def forward(self, query, key, value):
        assert query.dim() in [2, 3], ("query should be unbatched 2D or batched 3D tensor "
                                       f"but received {query.dim()}-D tensor")
        if query.dim() == 4:
            query = query.view(query.shape[0] * query.shape[1], query.shape[2], query.shape[3])

        batch_size, seq_len, dims = query.size()
        k = max(1, int(seq_len * self.sparsity_factor))
        indices = torch.topk(query.norm(dim=-1), k, dim=1).indices
        query_sparse = query.gather(1, indices.unsqueeze(-1).expand(-1, -1, dims))
        key_sparse = key.gather(1, indices.unsqueeze(-1).expand(-1, -1, dims))
        value_sparse = value.gather(1, indices.unsqueeze(-1).expand(-1, -1, dims))

        if query_sparse.shape[1] > 0 and key_sparse.shape[1] > 0 and value_sparse.shape[1] > 0:
            query_sparse = query_sparse.view(query_sparse.shape[0], query_sparse.shape[1], self.head, -1)
            key_sparse = key_sparse.view(key_sparse.shape[0], key_sparse.shape[1], self.head, -1)
            value_sparse = value_sparse.view(value_sparse.shape[0], value_sparse.shape[1], self.head, -1)

        attn_output, attn_weights = self.multihead_attn(query_sparse, key_sparse, value_sparse)
        return attn_output, attn_weights

class AdaptiveSpanAttention(nn.Module):
    def __init__(self, base, dims, head, max_dist, max_span):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dims, head)
        self.max_span = max_span
        self.span_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, query, key, value):
        span_length = int(self.max_span * self.span_scale.item())
        span_length = min(span_length, query.shape[1])
        query_span = query[:, :span_length, :]
        key_span = key[:, :span_length, :]
        value_span = value[:, :span_length, :]
        attn_output, attn_weights = self.multihead_attn(query_span, key_span, value_span)
        return attn_output, attn_weights

class RecurrentAttention(nn.Module):
    def __init__(self, base, dims, head, max_dist, chunk_size):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dims, head)
        self.chunk_size = chunk_size

    def forward(self, query, key, value, kv_cache=None):
        batch_size, seq_len, dims = query.size()
        output = torch.zeros_like(query).to(query.device)

        if kv_cache is None:
            kv_cache = {}
        key_global = key
        value_global = value

        for i in range(0, seq_len, self.chunk_size):
            end = min(seq_len, i + self.chunk_size)
            query_chunk = query[:, i:end, :]

            if 'k' not in kv_cache:
                kv_cache['k'] = key_global.clone().detach().to(query.device)
                kv_cache['v'] = value_global.clone().detach().to(query.device)

            key_chunk = kv_cache['k'][:, :end, :]
            value_chunk = kv_cache['v'][:, :end, :]

            attn_output, _ = self.multihead_attn(query_chunk, key_chunk, value_chunk)
            output[:, i:end, :] = attn_output
        return output, kv_cache

class CombinedAdaptiveSpanRecurrentAttention(nn.Module):
    def __init__(self, base, dims, head, max_dist, max_span, chunk_size):
        super().__init__()
        self.head = head
        self.multihead_attn = nn.MultiheadAttention(dims, head)
        self.max_span = max_span
        self.span_scale = nn.Parameter(torch.tensor(1.0))
        self.chunk_size = chunk_size

    def forward(self, query, key, value, kv_cache=None):
        assert query.dim() in [2, 3], ("query should be unbatched 2D or batched 3D tensor "
                                       f"but received {query.dim()}-D tensor")
        if query.dim() == 4:
            query = query.view(query.shape[0] * query.shape[1], query.shape[2], query.shape[3])

        batch_size, seq_len, dims = query.size()
        output = torch.zeros_like(query).to(query.device)

        if kv_cache is None:
            kv_cache = {}
        key_global = key
        value_global = value

        for i in range(0, seq_len, self.chunk_size):
            end = min(seq_len, i + self.chunk_size)
            query_chunk = query[:, i:end, :]

            if 'k' not in kv_cache:
                kv_cache['k'] = key_global.clone().detach().to(query.device)
                kv_cache['v'] = value_global.clone().detach().to(query.device)

            key_chunk = kv_cache['k'][:, :end, :]
            value_chunk = kv_cache['v'][:, :end, :]

            span_length = int(self.max_span * self.span_scale.item())
            span_length = min(span_length, query_chunk.shape[1])
            query_span = query_chunk[:, :span_length, :]
            key_span = key_chunk[:, :span_length, :]
            value_span = value_chunk[:, :span_length, :]

            attn_output, _ = self.multihead_attn(query_span, key_span, value_span)
            output[:, i:end, :] = attn_output

        return output, kv_cache

class HybridAttention(nn.Module):
    def __init__(self, base, dims, head, max_dist, win_size, max_span=64, sparsity_factor=0.1): 
        super().__init__()
        self.win_size=win_size
        self.local_attn = AdaptiveSpanAttention(base, dims, head, max_dist, max_span)
        # self.local_attn = SparseAttention(base, dims, head, max_dist, sparsity_factor)
        self.global_attn = MultiheadAttention(base, dims, head, max_dist)
        self.ln_local = LayerNorm(dims)
        self.ln_global = LayerNorm(dims)
        self.projection = Linear(2 * dims, dims)
        
        # self.combined_rotary = CombinedRotaryEmbedding(base, dims, head)

    def update_window(self, new_window):
        new_window = max(1, int(new_window + 0.5))
        win_size = new_window
        self.local_attn.max_span = win_size

    def forward(self, x): 
        win_size = self.win_size
        x_local = self.ln_local(x)
        x_global = self.ln_global(x)
        x_local = x_local.permute(1, 0, 2)  
        x_global = x_global.permute(1, 0, 2) 

        local_out = self.sliding_window_attention(x_local, win_size)
        # global_out, _ = self.global_attn(self.combined_rotary(x_global), self.combined_rotary(x_global), self.combined_rotary(x_global))
        global_out, _ = self.global_attn(x_global, x_global, x_global)

        if local_out.shape[1] != global_out.shape[1]:
            seq_len_diff = local_out.shape[1] - global_out.shape[1]
            if seq_len_diff > 0:
                local_out = local_out[:, :global_out.shape[1], :]
            elif seq_len_diff < 0:
                pad = (0, 0, 0, -seq_len_diff, 0, 0)
                local_out = F.pad(local_out, pad).to(local_out.device)

        combined = torch.cat([local_out, global_out], dim=-1)
        combined_out = self.projection(combined)
        return combined_out.permute(1, 0, 2) 

    def sliding_window_attention(self, x, win_size):
        batch_size, seq_len, dims = x.size()
        output = torch.zeros_like(x)

        for i in range(0, seq_len, win_size):
            end = min(i + win_size, seq_len)
            query = x[i:end, :, :]
            start = max(0, i - win_size)
            key = x[start:end, :, :]
            value = x[start:end, :, :]
            attn_output, _ = self.local_attn(query, key, value)
            output[i:end, :, :] = attn_output[:end - i, :, :]
        return output


###### Focused attention  WIP 

# class AdaptiveSpanAttention(nn.Module):
#     def __init__(self, base, dims, head, max_dist, win_size, max_span, temp_scale=0.01):
#         super().__init__()

#         self.max_dist = max_dist
#         self.win_size = win_size
#         self.max_span = max_span
#         self.temp_scale = temp_scale
#         self.multi_attn = MultiheadAttention(base, dims, head, max_dist)
#         self.span_scale = nn.Parameter(torch.tensor(1.0))

#     def forward(self, query, key, value, span_scale):
#         span_len = int(self.max_span * span_scale.mean().item())
#         span_len = min(span_len, query.shape[1], key.shape[1], value.shape[1])

#         eff_span = min(span_len, self.max_dist)
#         q_span = query[:, :eff_span, :]
#         k_span = key[:, :eff_span, :]
#         v_span = value[:, :eff_span, :]

#         attn_out, attn_weights = self.multi_attn(q_span, k_span, v_span)
#         temperature = 1.0 - self.temp_scale * span_scale  

#         n_batch, n_ctx, dims = q_span.shape
#         scale = (dims // self.multi_attn.head) ** -0.25

#         q = q_span.view(*q_span.shape[:2], self.multi_attn.head, -1).permute(0, 2, 1, 3)
#         k = k_span.view(*k_span.shape[:2], self.multi_attn.head, -1).permute(0, 2, 1, 3)
#         v = v_span.view(*v_span.shape[:2], self.multi_attn.head, -1).permute(0, 2, 1, 3)

#         attn_scores = torch.matmul(q, k.transpose(-2, -1))
#         attn_weights = torch.softmax((attn_scores / temperature) * scale, dim=-1)
#         attn_out = torch.matmul(attn_weights, v)
#         attn_out = attn_out.permute(0, 2, 1, 3).flatten(start_dim=2)
#         attn_weights = attn_weights * (1.0 / span_scale)     
        
#         attn_out = torch.bmm(attn_weights.view(-1, *attn_weights.shape[2:]), v.view(-1, *v.shape[2:]))
#         attn_out = attn_out.view(query.size(0), query.size(1), -1)
#         attn_out = attn_out.permute(0, 2, 1).contiguous().view(query.size(0), -1, query.size(2))    
#         # print(f"Adaptive {attn_out.shape}")
#         return attn_out, attn_weights

# class SpanPredictor(nn.Module):
#     def __init__(self, dims):
#         super().__init__()
#         self.linear = nn.Linear(dims, 1)

#     def forward(self, global_out):
#         scale = torch.sigmoid(self.linear(global_out))
#         return scale
    
# class HybridAttention(nn.Module):
#     def __init__(self, base, dims, head, max_dist, win_size = 32, max_span = 32, slid_win = 32):
#         super().__init__()
#         self.max_dist = max_dist
#         self.win_size = win_size
#         self.max_span = max_span
#         self.slid_win = slid_win

#         self.span_pred = SpanPredictor(dims)
#         self.dist_local = max_dist  
#         self.dist_global = max_dist
#         self.attn_local = AdaptiveSpanAttention(base, dims, head, self.dist_local, win_size, max_span)
#         self.attn_global = MultiheadAttention(base, dims, head, self.dist_global)
#         self.ln_local = LayerNorm(dims)
#         self.ln_global = LayerNorm(dims)
#         self.projection = Linear(2 * dims, dims)

#     def forward(self, x, new_dist=None, new_base=None, xa = None, mask = None, kv_cache = None):
    
#         local = self.ln_local(x)
#         globe= self.ln_global(x)

#         globe_out, _ = self.attn_global(globe, globe, globe)
#         span_scale = self.span_pred(globe_out.mean(dim=1)) 

#         win_size = max(1, int(self.slid_win * span_scale.mean().item()))
#         span_len = max(1, int(self.max_span * span_scale.mean().item()))

#         effective_max_dist = min(self.max_dist, local.size(1))
#         local_max_dist = min(self.dist_local, span_len, win_size)
#         globe_max_dist = effective_max_dist
#         self.attn_local.max_dist = local_max_dist
#         self.attn_global.max_dist = globe_max_dist

#         local_out = self.slid_wiattention(local, win_size, span_len, span_scale)
#         combined = torch.cat([local_out.permute(1, 0, 2), globe_out.permute(1, 0, 2)], dim=-1)
#         x = self.projection(combined)
#         return x
    
#     def slid_wiattention(self, x, win_size, span_len, span_scale):
#         batch_size, seq_len, dims = x.size()
#         out = torch.zeros_like(x, device=x.device)

#         for i in range(0, seq_len, win_size):
#             end = min(i + win_size, seq_len)
#             query = x[:, i:end, :]
#             start = max(0, i - span_len + win_size)
#             key = x[:, start:i + span_len, :]
#             value = x[:, start:i + span_len, :]
    
#             attn_out, _ = self.attn_local(query, key, value, span_scale)
#             x[:, i:end, :] = attn_out
#         # print(f"Hybrid {x.shape}")
#         return x
