
class HybridAttention(nn.Module):
    def __init__(self, base, n_state, n_head, max_rel_dist, window_size, max_span=64, loss=None): 
        super().__init__()
        assert n_state % n_head == 0, "n_state must be divisible by n_head"
        self.base = base
        self.loss = loss  
        self.max_span = max_span
        self.max_rel_dist = max_rel_dist
        self.n_state = n_state
        self.n_head = n_head
        self.window_size = window_size

        self.local_attn = AdaptiveSpanAttention(base, n_state, n_head, max_rel_dist, max_span)
        self.global_attn = MultiheadAttention(base, n_state, n_head, max_rel_dist, None)  
        self.ln_local = LayerNorm(n_state)
        self.ln_global = LayerNorm(n_state)

        self.projection = Linear(2 * n_state, n_state)

    def update_window(self, new_window):
        new_window = max(1, int(new_window + 0.5))
        self.window_size = new_window
        self.local_attn.max_span = new_window

    def forward(self, x, loss=None): 
        if loss is not None:
            self.update_window(loss) 
        window_size = self.window_size

        x_local = self.ln_local(x)
        x_global = self.ln_global(x)
        x_local = x_local.permute(1, 0, 2)  
        x_global = x_global.permute(1, 0, 2) 

        local_out = self.sliding_window_attention(x_local, window_size)
        # kv_cache = {}  
        global_out, _ = self.global_attn(x_global, x_global, x_global)#, kv_cache=kv_cache)

        if local_out.shape[1] != global_out.shape[1]:
            seq_len_diff = local_out.shape[1] - global_out.shape[1]
            if seq_len_diff > 0:
                local_out = local_out[:, :global_out.shape[1], :]
            elif seq_len_diff < 0:
                pad = (0, 0, 0, -seq_len_diff, 0, 0)
                local_out = F.pad(local_out, pad).to(local_out.device)

        combined = torch.cat([local_out, global_out], dim=-1)
        combined_out = self.projection(combined)
        return combined_out.permute(1, 0, 2)  # Permute back?

    def sliding_window_attention(self, x, window_size):
        batch_size, seq_len, n_state = x.size()
        output = torch.zeros_like(x)

        for i in range(0, seq_len, window_size):
            end = min(i + window_size, seq_len)
            query = x[i:end, :, :]
            start = max(0, i - window_size)
            key = x[start:end, :, :]
            value = x[start:end, :, :]
            attn_output, _ = self.local_attn(query, key, value)
            output[i:end, :, :] = attn_output[:end - i, :, :]
        return output


class MultiheadAttention(nn.Module):
    use_sdpa = True

    def __init__(self, base, n_state, n_head, max_rel_dist, window_size): 
        super().__init__()
        assert n_state % n_head == 0, "n_state must be divisible by n_head"
        self.n_head = n_head
        self.h_dim = n_state // n_head
        assert self.h_dim % 2 == 0, "Head dimension must be even for rotary embeddings"

        self.positional_scaling = nn.Parameter(torch.ones(1))
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)
        self.kv_cache = {}

        self.max_rel_dist = max_rel_dist
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.h_dim, 2).float() / self.h_dim))
        self.register_buffer('inv_freq', inv_freq)

        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * self.max_rel_dist - 1, self.n_head))

        self.combined_rotary = CombinedRotaryEmbedding(
            base=base,
            n_state=n_state, 
            n_head=n_head,  
            checkpointing=False 
        )

    def update_base(self, new_base):
        self.base = float(new_base)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.h_dim, 2).float() / self.h_dim))
        self.register_buffer('inv_freq', inv_freq)
        self.combined_rotary.update_base(self.base)

    def forward(self, x: Tensor, xa: Optional[Tensor] = None, 
                mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        q = self.combined_rotary(q) * self.positional_scaling
        k = self.combined_rotary(k) * self.positional_scaling

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, 
                      mask: Optional[Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, n_state = q.shape

        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if MultiheadAttention.use_sdpa:
            a = scaled_dot_product_attention(q, k, v, is_causal=mask is not None and n_ctx > 1)
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None 
        else:
            L, S = q.size(-2), k.size(-2)
            scale_factor = 1 / math.sqrt(q.size(-1)) if scale is None else scale
            attn_bias = torch.zeros(L, S, dtype=q.dtype)  
            w = q @ k.transpose(-2, -1) * scale_factor
            w += attn_bias.to(q.dtype).to(q.device)  
            w = torch.softmax(w, dim=-1).to(q.dtype)

            qk = (q * scale) @ (k * scale).transpose(-1, -2)

            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]

            qk = qk.float()
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()

        seq_len_q = q.size(2)
        seq_len_k = k.size(2)

        positions = (torch.arange(seq_len_q, device=q.device).unsqueeze(1) - 
                     torch.arange(seq_len_k, device=q.device).unsqueeze(0))
        positions = positions.clamp(min=-self.max_rel_dist + 1, max=self.max_rel_dist - 1) + self.max_rel_dist - 1
        rel_bias = self.rel_pos_bias[positions]
        rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)
        qk = qk + rel_bias if qk is not None else rel_bias  
        return out, qk

class CombinedSparseAdaptiveAttention(nn.Module):
    def __init__(self, base, n_state, n_head, max_rel_dist, max_span, sparsity_factor):
        super().__init__()
        self.n_head = n_head
        self.multihead_attn = MultiheadAttention(base, n_state, n_head, max_rel_dist, max_span) 
        self.sparsity_factor = sparsity_factor
        self.max_span = max_span
        self.span_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, query, key, value): 
        assert query.dim() in [2, 3], ("query should be unbatched 2D or batched 3D tensor "
                                       f"but received {query.dim()}-D tensor")
        if query.dim() == 4:
            query = query.view(query.shape[0] * query.shape[1], query.shape[2], query.shape[3])

        batch_size, seq_len, n_state = query.size()
        k = max(1, int(seq_len * self.sparsity_factor))
        indices = torch.topk(query.norm(dim=-1), k, dim=1).indices
        query_sparse = query.gather(1, indices.unsqueeze(-1).expand(-1, -1, n_state))
        key_sparse = key.gather(1, indices.unsqueeze(-1).expand(-1, -1, n_state))
        value_sparse = value.gather(1, indices.unsqueeze(-1).expand(-1, -1, n_state))

        if query_sparse.shape[1] > 0 and key_sparse.shape[1] > 0 and value_sparse.shape[1] > 0: 
            query_sparse = query_sparse.view(query_sparse.shape[0], query_sparse.shape[1], self.n_head, -1)
            key_sparse = key_sparse.view(key_sparse.shape[0], key_sparse.shape[1], self.n_head, -1)
            value_sparse = value_sparse.view(value_sparse.shape[0], value_sparse.shape[1], self.n_head, -1)

        span_length = int(self.max_span * self.span_scale.item())
        span_length = min(span_length, query.shape[1])
        query_span = query_sparse[:, :span_length, :]
        key_span = key_sparse[:, :span_length, :]
        value_span = value_sparse[:, :span_length, :]

        attn_output, attn_weights = self.multihead_attn(query_span, key_span, value_span) 
        return attn_output, attn_weights

class CombinedSparseAdaptiveAttention(nn.Module):
    def __init__(self, base, n_state, n_head, max_rel_dist, max_span, sparsity_factor):
        super().__init__()
        self.n_head = n_head
        self.multihead_attn = MultiheadAttention(base, n_state, n_head, max_rel_dist, max_span)  
        self.sparsity_factor = sparsity_factor
        self.max_span = max_span
        self.span_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, query, key, value): 
        assert query.dim() in [2, 3], ("query should be unbatched 2D or batched 3D tensor "
                                       f"but received {query.dim()}-D tensor")
        if query.dim() == 4:
            query = query.view(query.shape[0] * query.shape[1], query.shape[2], query.shape[3])

        batch_size, seq_len, n_state = query.size()
        k = max(1, int(seq_len * self.sparsity_factor))
        indices = torch.topk(query.norm(dim=-1), k, dim=1).indices
        query_sparse = query.gather(1, indices.unsqueeze(-1).expand(-1, -1, n_state))
        key_sparse = key.gather(1, indices.unsqueeze(-1).expand(-1, -1, n_state))
        value_sparse = value.gather(1, indices.unsqueeze(-1).expand(-1, -1, n_state))

        if query_sparse.shape[1] > 0 and key_sparse.shape[1] > 0 and value_sparse.shape[1] > 0: 
            query_sparse = query_sparse.view(query_sparse.shape[0], query_sparse.shape[1], self.n_head, -1)
            key_sparse = key_sparse.view(key_sparse.shape[0], key_sparse.shape[1], self.n_head, -1)
            value_sparse = value_sparse.view(value_sparse.shape[0], value_sparse.shape[1], self.n_head, -1)

        span_length = int(self.max_span * self.span_scale.item())
        span_length = min(span_length, query.shape[1])
        query_span = query_sparse[:, :span_length, :]
        key_span = key_sparse[:, :span_length, :]
        value_span = value_sparse[:, :span_length, :]

        attn_output, attn_weights = self.multihead_attn(query_span, key_span, value_span) 
        return attn_output, attn_weights

class SparseAttention(nn.Module):
    def __init__(self, base, n_state, n_head, max_rel_dist, sparsity_factor):
        super().__init__()
        self.n_head = n_head
        self.multihead_attn = MultiheadAttention(base, n_state, n_head, max_rel_dist, None)  
        self.sparsity_factor = sparsity_factor

    def forward(self, query, key, value):
        assert query.dim() in [2, 3], ("query should be unbatched 2D or batched 3D tensor "
                                       f"but received {query.dim()}-D tensor")
        if query.dim() == 4:
            query = query.view(query.shape[0] * query.shape[1], query.shape[2], query.shape[3])

        batch_size, seq_len, n_state = query.size()
        k = max(1, int(seq_len * self.sparsity_factor))
        indices = torch.topk(query.norm(dim=-1), k, dim=1).indices
        query_sparse = query.gather(1, indices.unsqueeze(-1).expand(-1, -1, n_state))
        key_sparse = key.gather(1, indices.unsqueeze(-1).expand(-1, -1, n_state))
        value_sparse = value.gather(1, indices.unsqueeze(-1).expand(-1, -1, n_state))

        if query_sparse.shape[1] > 0 and key_sparse.shape[1] > 0 and value_sparse.shape[1] > 0:
            query_sparse = query_sparse.view(query_sparse.shape[0], query_sparse.shape[1], self.n_head, -1)
            key_sparse = key_sparse.view(key_sparse.shape[0], key_sparse.shape[1], self.n_head, -1)
            value_sparse = value_sparse.view(value_sparse.shape[0], value_sparse.shape[1], self.n_head, -1)

        attn_output, attn_weights = self.multihead_attn(query_sparse, key_sparse, value_sparse)
        return attn_output, attn_weights

class AdaptiveSpanAttention(nn.Module):
    def __init__(self, base, n_state, n_head, max_rel_dist, max_span):
        super().__init__()
        self.multihead_attn = MultiheadAttention(base, n_state, n_head, max_rel_dist, None) 
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
    def __init__(self, base, n_state, n_head, max_rel_dist, chunk_size):
        super().__init__()
        self.multihead_attn = MultiheadAttention(base, n_state, n_head, max_rel_dist, None) 
        self.chunk_size = chunk_size

    def forward(self, query, key, value, kv_cache=None):
        batch_size, seq_len, n_state = query.size()
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
    def __init__(self, base, n_state, n_head, max_rel_dist, max_span, chunk_size):
        super().__init__()
        self.n_head = n_head
        self.multihead_attn = MultiheadAttention(base, n_state, n_head, max_rel_dist, None)  
        self.max_span = max_span
        self.span_scale = nn.Parameter(torch.tensor(1.0))
        self.chunk_size = chunk_size

    def forward(self, query, key, value, kv_cache=None):
        assert query.dim() in [2, 3], ("query should be unbatched 2D or batched 3D tensor "
                                       f"but received {query.dim()}-D tensor")
        if query.dim() == 4:
            query = query.view(query.shape[0] * query.shape[1], query.shape[2], query.shape[3])

        batch_size, seq_len, n_state = query.size()
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
