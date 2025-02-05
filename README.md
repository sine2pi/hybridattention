    
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
