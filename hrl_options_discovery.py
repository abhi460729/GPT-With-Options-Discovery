import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from gym import spaces
from sklearn.cluster import KMeans
import random
from collections import deque
import asyncio
import platform

# Hyperparameters for GPT
batch_size = 64
block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
learning_rate = 3e-4
max_iters = 5000
eval_interval = 500
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters for HRL
state_dim = n_embd  # State is the embedding of the current sequence
action_dim = 10  # Number of high-level options (to be discovered)
hidden_dim = 64
lambda_intrinsic = 0.1
episodes = 1000
max_steps = 100

torch.manual_seed(1337)

# Load and preprocess text data (same as original GPT code)
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading for GPT
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# GPT Model Components (same as original)
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss, x  # Return embeddings for HRL

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Text Generation Environment
class TextGenerationEnv(gym.Env):
    def __init__(self, gpt_model):
        super(TextGenerationEnv, self).__init__()
        self.gpt_model = gpt_model
        self.action_space = spaces.Discrete(action_dim)  # High-level options
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(n_embd,), dtype=np.float32)
        self.state = None
        self.sequence = None
        self.current_step = 0
        self.max_steps = max_steps

    def reset(self):
        self.sequence = torch.zeros((1, 1), dtype=torch.long, device=device)
        _, _, embedding = self.gpt_model(self.sequence)
        self.state = embedding[:, -1, :].detach().cpu().numpy().flatten()
        self.current_step = 0
        return self.state

    def step(self, action):
        # Map action (option) to a sequence of token generations
        option_length = 5  # Number of tokens per option
        current_sequence = self.sequence
        for _ in range(option_length):
            logits, _, embedding = self.gpt_model(current_sequence[:, -block_size:])
            logits = logits[:, -1, :]
            # Bias logits towards a specific style based on action (simplified)
            logits = logits + torch.randn_like(logits) * 0.1 * action
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            current_sequence = torch.cat((current_sequence, next_token), dim=1)
        self.sequence = current_sequence
        self.state = embedding[:, -1, :].detach().cpu().numpy().flatten()
        self.current_step += 1

        # Reward: 1.0 if sequence length reaches a threshold, else 0
        reward = 1.0 if self.sequence.shape[1] >= 50 else 0.0
        done = self.current_step >= self.max_steps or self.sequence.shape[1] >= 50
        return self.state, reward, done, {'sequence': decode(self.sequence[0].tolist())}

    def render(self):
        print(decode(self.sequence[0].tolist()))

# Intrinsic Curiosity Module (ICM)
class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ICM, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        self.inverse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, state, next_state, action):
        state_f = self.feature(state)
        next_state_f = self.feature(next_state)
        pred_action = self.inverse(torch.cat([state_f, next_state_f], dim=-1))
        pred_next_f = self.forward_model(torch.cat([state_f, action], dim=-1))
        return pred_next_f, pred_action

# Simplified PPO Policy
class PPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PPOPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

# Training Loop
async def train_gpt_hrl():
    gpt_model = GPTLanguageModel().to(device)
    env = TextGenerationEnv(gpt_model)
    icm = ICM(state_dim, action_dim).to(device)
    policy = PPOPolicy(state_dim, action_dim).to(device)
    gpt_optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=learning_rate)
    icm_optimizer = torch.optim.Adam(icm.parameters(), lr=0.001)
    policy_optimizer = torch.optim.Adam(policy

.parameters(), lr=0.001)
    buffer = deque(maxlen=10000)
    trajectories = []

    # Train GPT model first
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(gpt_model)
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch('train')
        logits, loss, _ = gpt_model(xb, yb)
        gpt_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gpt_optimizer.step()

    # HRL Training
    for episode in range(episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        episode_traj = []
        total_reward = 0
        done = False
        step = 0

        while not done and step < env.max_steps:
            probs = policy(state)
            action = torch.multinomial(probs, 1).item()
            next_state, reward, done, info = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            episode_traj.append((state.clone(), action, next_state.clone()))

            action_onehot = torch.zeros(1, action_dim, device=device)
            action_onehot[0, action] = 1
            pred_next_f, pred_action = icm(state, next_state, action_onehot)
            intrinsic_reward = torch.norm(pred_next_f - icm.feature(next_state), dim=-1).pow(2).detach()
            total_reward += reward + lambda_intrinsic * intrinsic_reward.item()

            buffer.append((state.clone(), action, reward, next_state.clone(), intrinsic_reward.clone()))

            if len(buffer) > 32:
                batch = random.sample(buffer, 32)
                states, actions, rewards, next_states, intrinsic_rewards = zip(*batch)
                states = torch.cat(states)
                next_states = torch.cat(next_states)
                actions = torch.tensor(actions, device=device)
                action_onehot = torch.zeros(len(actions), action_dim, device=device)
                action_onehot.scatter_(1, actions.unsqueeze(1), 1)
                intrinsic_rewards = torch.cat(intrinsic_rewards)

                # Policy loss
                policy_optimizer.zero_grad()
                probs = policy(states)
                log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)))
                policy_loss = -(log_probs * (torch.tensor(rewards, device=device) + lambda_intrinsic * intrinsic_rewards)).mean()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                policy_optimizer.step()

                # ICM loss
                icm_optimizer.zero_grad()
                pred_next_f, pred_action = icm(states, next_states, action_onehot)
                icm_loss = torch.norm(pred_next_f - icm.feature(next_states), dim=-1).pow(2).mean() + \
                           nn.CrossEntropyLoss()(pred_action, actions)
                icm_loss.backward()
                torch.nn.utils.clip_grad_norm_(icm.parameters(), max_norm=1.0)
                icm_optimizer.step()

            state = next_state
            step += 1

        trajectories.append(episode_traj)
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Generated: {info['sequence'][:50]}...")

    # Cluster trajectories to discover options
    states = np.concatenate([np.array([s.cpu().numpy() for s, _, _ in traj]) for traj in trajectories])
    kmeans = KMeans(n_clusters=action_dim)
    option_labels = kmeans.fit_predict(states)
    print("Option labels:", np.unique(option_labels, return_counts=True))

    # Generate sample text
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = decode(gpt_model.generate(context, max_new_tokens=500)[0].tolist())
    print("Sample Generated Text:\n", generated)

if platform.system() == "Emscripten":
    asyncio.ensure_future(train_gpt_hrl())
else:
    if __name__ == "__main__":
        asyncio.run(train_gpt_hrl())
