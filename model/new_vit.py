import torch
import torch.nn as nn



class VitEmbeddings(nn.Module):
    def __init__(self, img_size, patch_size, hidden_size):
        super(VitEmbeddings, self).__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, hidden_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.num_patches = self.patch_embedding.num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_size))
        self.patch_size = patch_size

    def forward(self, img):
        batch_size, num_channels, height, width = img.shape
        embeddings = self.patch_embedding(img)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)
        embeddings = embeddings + self.pos_embed
        return embeddings

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, hidden_size):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) # shape : (B, E, n, n) where B is batch size, E is embed dim, n is number of patches in row or column
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class VitSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(VitSelfAttention, self).__init__()
        self.key = nn.Linear(hidden_size, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attention_head_size = hidden_size // num_heads
        self.num_heads = num_heads
        assert self.attention_head_size * num_heads == hidden_size, "hidden size must be divisible by num_heads"

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        batch_size, num_patches, hidden_size = x.shape
        key_layer = self.transpose_for_scores(self.key(x))
        query_layer = self.transpose_for_scores(self.query(x))
        value_layer = self.transpose_for_scores(self.value(x))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (hidden_size ** 0.5)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).reshape(batch_size, num_patches, hidden_size)
        return context_layer

class VitAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(VitAttention, self).__init__()
        self.self_attention = VitSelfAttention(hidden_size, num_heads)
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        attention_output = self.self_attention(x)
        attention_output = self.dense(attention_output)
        return attention_output

class VitIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(VitIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_size = intermediate_size

    def forward(self, x):
        x = self.dense(x)
        x = nn.GELU()(x)
        return x

class VitOutput(nn.Module):
    def __init__(self, intermediate_size, hidden_size):
        super(VitOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states

class VitLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super(VitLayer, self).__init__()
        self.layer_norm_before = nn.LayerNorm(hidden_size)
        self.attention = VitAttention(hidden_size, num_heads)
        self.layer_norm_after = nn.LayerNorm(hidden_size)
        self.intermediate = VitIntermediate(hidden_size, intermediate_size)
        self.output = VitOutput(intermediate_size, hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.layer_norm_before(hidden_states)
        attention_output = self.attention(hidden_states)
        hidden_states = attention_output + hidden_states
        hidden_states = self.layer_norm_after(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, hidden_states)
        return layer_output

class VitEncoder(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, num_layers):
        super(VitEncoder, self).__init__()
        self.layer_module = nn.ModuleList([VitLayer(hidden_size=hidden_size, num_heads=num_heads, intermediate_size=intermediate_size) for _ in range(num_layers)])

    def forward(self, hidden_states):
        for layer_module in self.layer_module:
            hidden_states = layer_module(hidden_states)

        return hidden_states

class VitModel(nn.Module):
    def __init__(self, img_size, patch_size, hidden_size, num_heads, intermediate_size, num_layers):
        super(VitModel, self).__init__()
        self.embeddings = VitEmbeddings(img_size, patch_size, hidden_size)
        self.encoder = VitEncoder(hidden_size, num_heads, intermediate_size, num_layers)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, img):
        hidden_states = self.embeddings(img)
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states

class VitClassifier(nn.Module):
    def __init__(self, img_size, patch_size, num_heads, intermediate_size, num_layers, hidden_size, num_classes):
        super(VitClassifier, self).__init__()
        self.vit = VitModel(img_size, patch_size, hidden_size, num_heads, intermediate_size, num_layers)
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, img):
        hidden_states = self.vit(img)
        logits = self.dense(hidden_states[:, 0, :])
        return logits