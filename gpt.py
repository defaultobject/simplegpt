import jax
import jax.numpy as np
from jax import nn
import chex
import objax
import numpy as onp

from tqdm import trange

from jax import jit
from functools import partial

@jit
def NewGELU(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.pow(x, 3.0))))

# Functionals
@partial(jit, static_argnums=(2, 3))
def dropout(A, p, generator, shape=None):

    if True:
        # DEBUGGING
        return A

    if shape is None:
        shape = [A.shape[0], A.shape[1]]

    uni_mask = objax.random.uniform(shape, generator)
    dropout_mask = np.where(uni_mask < p, 0.0, 1.0)

    # (?) Why scaled?
    dropout_mask = dropout_mask * (1/(1-p))

    #return A
    return A*dropout_mask

@partial(jit, static_argnums=(3))
def layer_norm(A, gamma, beta, axis):
    eps = 1e-15 # to avoid divide by zeros

    mu = np.mean(A, axis=axis)[:, None]
    var = np.var(A,axis=axis)[:, None]

    return ((A-mu)/np.sqrt(var+eps))*gamma + beta

@jit
def linear_layer(X, W):
    """
    f: R^N -> R^M
    
    Args:
        X: (T x N)
        W: (N x M)
        
    Output:
        Y: (T x M)
    """
    chex.assert_rank([X, W], [2, 2])
    chex.assert_equal(X.shape[1], W.shape[0])

    return X @ W

@jit
def linear_layer_with_bias(X, W, B):
    """
    f: R^N -> R^M
    
    Args:
        X: (T x N)
        W: (N x M)
        
    Output:
        Y: (T x M)
    """
    lin_proj =  linear_layer(X, W) 

    chex.assert_equal(lin_proj.shape[1], B.shape[0])

    return lin_proj + B



@partial(jit, static_argnums=(3), static_argnames=['mask', 'scale', 'apply_dropout', 'dropout_p', 'verbose'])
def dot_product_self_attention(Q, K, V, generator, mask=True, scale=True, apply_dropout=True, dropout_p = 0.1, verbose = False):
    """
    Notation:
        T: sequence length
        C: embedding dimenson
    Args:
        Q (T x Cq): Query matrix
        K (T x Ck): Key matrix
        V (T x Cv): Value matrix
        generator: Random number generator. Required for dropout
        mask (bool): if True compute causal/masked self-attention
        scale (bool): if True scale by 1/sqrt(Dk)
        apply_dropout (bool): if True apply dropout to attention matrix
        dropout_p(float): probablity of dropout
    """
    # Extract shapes
    T, Cq = Q.shape
    Ck = K.shape[1]
    Cv = V.shape[1]

    # Computing scaling factor
    if scale:    
        # (?) why use a scaling factor?
        # TBD
        
        scaling_factor = 1.0/np.sqrt(Cq)
        if verbose:
            print('G before scaling')
            print(Q @ K.T)
            print('scaling_factor: ', scaling_factor)
    else:
        scaling_factor = 1.0
        
    # Compute `gram matrix' - O(T^2)
    G = Q @ K.T * scaling_factor
    chex.assert_shape(G, [T, T])
    
    if verbose:
        print('Computing G')
        print(G)

    # compute attention matrix - O(T^2)
    

    if mask:
        # (?) what is the causal mask?
        # The attention matrix is `making information links` between all T tokens.
        #   For computational reasons we only want to information to flow from preceding tokens
        #   By masking the attention matrix to set the upper triangular (offset by 1) this stops the information
        #    from flowing `forward' .
        G = G.at[np.triu_indices(T, 1)].set(-np.inf)
        if verbose:
            print('Applying causal mask')
            print(G)

    # (?) why axis = 1?
    # We are computing Y = A X , we want the rows to be normalised such that
    #   y_n = \sum_m a_{nm} x_m with \sum_m a_{nm} = 1
    # hence use axis=1
    
    # (?) why softmax?
    # softmax(A, axis=1)_nm = exp(A_nm) / \sum exp(A_{n,i}) 
    # which guarentees that each element is positive and sums to 1 along axis 1 (rows)
            
    A =  nn.softmax(G, axis=1) 
    
    chex.assert_shape(A, [T, T])
    if verbose:
        print('Computing Attention Matrix')
        print(A)
        
    # if mask = True, compute causal attention

    if apply_dropout:
        A = dropout(A, dropout_p, generator, shape=(128, 128))
        if verbose:
            print('Applying dropout')
            print(A)
            
    # compute attention - O(T^2)
    V_ = A @ V
    chex.assert_equal_shape([V_, V])

    if verbose:
        print('Computing Attention')
        print(V_)

    return V_


@jit
def cross_entropy(logits, targets):
    """
    Args:
        logits: T x N
        targets: T
    """
    T, N = logits.shape
    one_hot = nn.one_hot(targets, N)
    #return objax.functional.loss.cross_entropy_logits(logits, one_hot)

    log_ss = nn.log_softmax(logits, axis=1)
    chex.assert_equal_shape([log_ss, one_hot])

    loss = -log_ss * one_hot 
    loss = loss.sum() / one_hot.sum() 
    return loss 

# Classes
class Sequential(objax.Module):
    def __init__(self, seq_list: list):
        self.seq_list = objax.ModuleList(seq_list)
        
    def forward(self, X):
        X_ = X
        for m in self.seq_list:
            X_ = m(X_)
        return X_
    
    def __getitem__(self,index):
         return self.seq_list[index]
    __call__ = forward
    
class Dropout(objax.Module):
    def __init__(self, p = 0.1, seed=0, generator=None):
        self.p = p
        if generator is None:
            self.generator = objax.random.Generator(seed=seed)
        else:
            self.generator = generator
            
    def forward(self, X):
        return X
        return dropout(X, self.p, self.generator)
    __call__ = forward


    
class Embedding(objax.Module):
    def __init__(self, dims: list, W = None):
        self.dims = dims
        
        if W is None:
            W = np.array(onp.random.randn(*dims))
            #W = np.array(np.ones(dims))
        else:
            W = np.array(W)
        self._W = objax.TrainVar(W)
        
    @property
    def W(self):
        return self._W.value
    
    def forward(self, X):
        return self.W[X]
        
    __call__ = forward
    

class PositionalEmedding(objax.Module):
    def __init__(self, dims: list, W = None):
        self.vocab_size = dims[0]
        self.embedding_dim = dims[1]
        self.context_size = dims[2]
        self.token_embedding = Embedding([dims[0], dims[1]], W = W)
        self.position_embedding = Embedding([dims[2], dims[1]], W = W)
        
    def forward(self, X):
        T = X.shape[0]
        pos = np.arange(T)[:, None]
        return self.token_embedding(np.squeeze(X)) + self.position_embedding(np.squeeze(pos))
    __call__ = forward

class Linear(objax.Module):
    def __init__(self, dims, W=None, bias=False, B = None, seed=0):
        self.dims = dims
        
        std = np.sqrt(1/dims[0])
        
        if W is None:
            W = np.array(onp.random.uniform(-std, std, dims))
        else:
            W = np.array(W)
            
        self._W = objax.TrainVar(W)
        
        self.include_bias = bias
        
        if bias:
            if B is None:
                B = np.array(onp.random.uniform(-std, std, dims[1]))
            else:
                B = np.array(B)
                
            self._B = objax.TrainVar(B)
        else:
            self._B = None
    
    @property
    def W(self):
        return self._W.value
    @property
    def B(self):
        return self._B.value
    
    def forward(self, X):
        if self.include_bias:
            return linear_layer_with_bias(X, self.W, self.B)
        else:
            return linear_layer(X, self.W)
    
    __call__ = forward
    
class MLP(objax.Module):
    def __init__(self, dims, seed=0):

        self.learnables = {
            'lin_1': Linear([dims[0], dims[1]], seed=0, bias=True),
            'lin_2': Linear([dims[1], dims[2]], seed=0, bias=True),
        }

        self.blocks = Sequential([
            self.learnables['lin_1'],
            NewGELU,
            self.learnables['lin_2']
        ])
        
    def forward(self, X):
        return self.blocks.forward(X)
    __call__ = forward
    
class Attention(objax.Module):
    def __init__(self, T, C, Cq, Ck, Cv, h, lin_mh_att, seed=0, generator=None):
        self. T = T # sequence length
        self.C = C # token embedding dimension
        self.Cq = Cq # query embedding dimension
        self.Ck = Ck # key embedding dimension
        self.Cv = Cv # value embedding dimension
        
        self.h = h
        self.lin_mh_att = lin_mh_att
        
        if generator is None:
            self.generator = objax.random.Generator(seed=seed)
        else:
            self.generator = generator
        
    def Q(self, X):
        return self.lin_mh_att(X)[:, :self.C][:, self.h*self.Cq:(self.h+1)*self.Cq]
    def K(self, X):
        return self.lin_mh_att(X)[:, self.C:self.C*2][:, self.h*self.Ck:(self.h+1)*self.Ck]
    def V(self, X):
        return self.lin_mh_att(X)[:, self.C*2:][:, self.h*self.Cv:(self.h+1)*self.Cv]
    
    def forward(self, X):

        return dot_product_self_attention(
            self.Q(X),
            self.K(X),
            self.V(X),
            self.generator
        )
    __call__ = forward

class BatchedMultiHeadAttention(objax.Module):
    def __init__(self, Nh, T, C, Cq, Ck, Cv, generator=None):
        # ASSERT all equal [Cq, Ck, Cv])

        self.T = T # token length
        self.Nh = Nh # number of heads
        self.C = C # token dim
        self.Cq = Cq 
        self.Ck = Ck 
        self.Cv = Cv 
        
        self.generator = generator
        
        self.lin_mh_att = Linear([self.C, Nh*(Cq+Ck+Cv)], bias=True)
        
        self.output_proj = Linear([Cv*Nh, C], bias=True)
        
    def forward(self, X):
        T_ = X.shape[0] # current token size
        lin_mh = self.lin_mh_att(X)
        lin_mh =  lin_mh.reshape([T_, 3, self.Nh, -1]) #[T, [QKV], Nh, Ck]

        # rearrange into batched Q, K , V
        Q = lin_mh[:, 0, ...]
        K = lin_mh[:, 1, ...]
        V = lin_mh[:, 2, ...]

        att_output = jax.vmap(
            lambda q, k, v, g: dot_product_self_attention(q, k, v, g),
            [1, 1, 1, None]
        )(Q, K, V, self.generator)
            
        # rearrange to match minGPT
        mh_att_output = np.reshape(np.transpose(np.stack(att_output), [1, 0, 2]), [X.shape[0], -1])
        chex.assert_shape(mh_att_output, [X.shape[0], self.Cv*self.Nh])
        out =  self.output_proj(mh_att_output)
        return out
    
    __call__ = forward

class MultiHeadAttention(objax.Module):
    def __init__(self, Nh, T, C, Cq, Ck, Cv, generator=None):
        self.T = T # token length
        self.Nh = Nh # number of heads
        self.C = C # token dim
        self.Cv = Cv # attention output_dim
        
        self.generator = generator
        
        self.lin_mh_att = Linear([self.C, Nh*(Cq+Ck+Cv)], bias=True)
        
        if True:
            self.attention_list = objax.ModuleList([
                Attention(T, C, Cq, Ck, Cv, nh, self.lin_mh_att, generator=generator)
                for nh in range(Nh)
            ])
        else:
            self.attention_list = batchjax.Batched([
                Attention(T, C, Cq, Ck, Cv, nh, self.lin_mh_att, generator=generator)
                for nh in range(Nh)
            ])

        self.output_proj = Linear([Cv*Nh, C], bias=True)
        
    def forward(self, X):
        if True:
            att_output = []
            for att in self.attention_list:
                att_output.append(att(X))

        else:
            att_output = batchjax.batch_or_loop(
                lambda att: att(X),
                inputs = [self.attention_list],
                axes=[0],
                dim = self.Nh,
                out_dim = 1,
                batch_type = batchjax.BatchType.BATCHED
            )
            
        # rearrange to match minGPT
        mh_att_output = np.reshape(np.transpose(np.stack(att_output), [1, 0, 2]), [X.shape[0], -1])
        chex.assert_shape(mh_att_output, [X.shape[0], self.Cv*self.Nh])
        out =  self.output_proj(mh_att_output)
        return out
    
    __call__ = forward
    
class Residual(objax.Module):
    def __init__(self, parent):
        self.parent = parent
    def forward(self, X):
        parent_out = self.parent(X)
        
        chex.assert_equal_shape([X, parent_out])
        return X+parent_out
    __call__ = forward

class LayerNorm(objax.Module):
    def __init__(self, dim, gamma = 1.0, beta=0.0):
        if True:
            self._gamma = objax.TrainVar(np.ones(dim)*np.array(gamma))
            self._beta = objax.TrainVar(np.ones(dim)*np.array(beta))
        else:
            self._gamma = objax.TrainVar(np.ones(1)*np.array(gamma))
            self._beta = objax.TrainVar(np.ones(1)*np.array(beta))
        
    @property
    def gamma(self):
        return self._gamma.value
    
    @property
    def beta(self):
        return self._beta.value
    
    def forward(self, X):
        return layer_norm(X, self.gamma, self.beta, axis=1)

    __call__ = forward


class Transformer(objax.Module):
    def __init__(self, Nh, T, C, Cq, Ck, Cv, generator=None):
        self.generator = generator

        self.learnables = {
            'ln_1': LayerNorm(C),
            'att': BatchedMultiHeadAttention(Nh, T, C, Cq, Ck, Cv, generator=generator),
            'ln_2': LayerNorm(C),
            'mlp': MLP([C, 4*C, C])
        }
        self.blocks = Sequential([
            Residual(
                Sequential([
                    self.learnables['ln_1'],
                    self.learnables['att'],
                ])
            ),
            Residual(
                Sequential([
                    self.learnables['ln_2'],
                    self.learnables['mlp']
                ])
            ),
            #LayerNorm(C),
            Dropout(generator=generator),
        ])

    def forward(self, X):
        return self.blocks(X)
    
    __call__ = forward 
    
class GPT(objax.Module):
    def __init__(self, N, T, C, Nh, Nl, seed=0):
        self.N = N # vocab size
        self.T = T # sequence length
        self.C = C # token embedding dimension
        self.Nh = Nh # number of attention heads
        self.Nl = Nl # number of layers/transformer blocks
        self.generator = objax.random.Generator(seed=seed)
        
        Cq = C//self.Nh
        Ck = C//self.Nh
        Cv = C//self.Nh
        

        self.blocks = Sequential([
            PositionalEmedding([N, C, T]),
            Dropout(generator=self.generator),
            Sequential([
                Transformer(Nh, T, C, Cq, Ck, Cv, generator=self.generator)
                for l in range(Nl)
            ]),
            LayerNorm(C),
            Linear(dims=[C, N], bias=False)
        ])
        
    def _blocks(self, X):
        if len(X.shape)==1:
            X = X[:, None]

        logits = self.blocks(X)
        return logits

    def forward(self, X):
        return jax.vmap(self._blocks)(X)

    def predict(self, X):
        X_ = self._blocks(X)
        Y =  nn.softmax(X_, axis=1)
        return Y.argmax(axis=1)

    def _objective(self, x, t):
        logits = self._blocks(x)
        return cross_entropy(logits, t)
    
    def objective(self, X, T):
        return np.sum(jax.vmap(self._objective, [0, 0])(X, T))
    
    __call__ = forward 

class Tokenizer(objax.Module):
    pass

class SimpleTokenizer(Tokenizer):
    """ https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py """
    def __init__(self):
        self.chars = None
        self.vocab_size = None
        self.str2int: dict = None
        self.int2str: dict = None
    def train(self, data):
        self.chars = sorted(list(set(data)))
        self.vocab_size = len(self.chars)
        self.str2int = { ch:i for i,ch in enumerate(self.chars) }
        self.int2str = { i:ch for i,ch in enumerate(self.chars) }
    def encode(self, txt):
        return [self.str2int[c] for c in txt]
    def decode(self, arr):
        return ''.join([self.int2str[i] for i in arr])
        
        

# Data Class
class Data(object):
    def __init__(self, data, block_size, batch_size, seed=0):
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size
        onp.random.seed(seed)
        
    def batch(self):
        ix = onp.random.randint(0, len(self.data) - self.block_size, self.batch_size)

        if False:
            # debugging
            ix = [10 for i in range(self.batch_size)]

        x = np.array([self.data[i:i+self.block_size] for i in ix])
        y = np.array([self.data[i+1:i+1+self.block_size] for i in ix])
        return x, y
# Trainers

def progress_bar_callback(num_epochs):
    """
        Simple progressbar - does not display learning objective value
    """
    from tqdm import tqdm


    bar = tqdm(total=num_epochs)

    def inner(epoch, grad, val):
        bar.update(1)

    return inner


def progress_bar_callback_notebook(num_epochs):
    """
        Simple progressbar - does not display learning objective value
    """
    from tqdm.notebook import trange, tqdm


    bar = tqdm(total=num_epochs)

    def inner(epoch, grad, val):
        bar.update(1)

    return inner

class GradDescentTrainer(object):
    def __init__(self, m, optimizer):
        self.obj_fn = objax.Jit(m.objective, m.vars())
        self.grad_fn = objax.Grad(self.obj_fn, m.vars())
        self.optimizer = optimizer
        self.opt = self.optimizer(m.vars())
        
    def train(
        self,
        data,
        learning_rate,
        epochs,
        callback=None,
    ):
        lc_arr = []

        def train_op():
            x,t = data.batch()
            grad = self.grad_fn(x,t)
            val = self.obj_fn(x,t)
            self.opt(learning_rate, grad)
            return grad, val

        for i in range(epochs):
            grad, val = train_op()

            if callback is not None:
                callback(i, grad, val)

            # Clean up val
            lc_arr.append(np.array(val).flatten())

        return np.array(lc_arr).flatten()
    
class ADAM(GradDescentTrainer):
    """ Constructs a GradDescentTrainer using Adam argument """
    def __init__(self, *args, **kwargs):
        super(ADAM, self).__init__(*args, optimizer=objax.optimizer.Adam, **kwargs)

def sample(prompt, N, gpt, tokenizer, seed=0, encode=True):
    gen = objax.random.Generator(seed=seed)
    
    if encode:
        prompt_enc = np.array(tokenizer.encode(prompt))
    else:
        prompt_enc = prompt

    res = prompt_enc

    # add padding so we dont have to rejit for every sample
    padding_required = max(gpt.T - res.shape[0], 0)


    for i in trange(N):
        # ensure block size
        if prompt_enc.shape[0] > gpt.T:
            prompt_enc = prompt_enc[-gpt.T:]

        elif padding_required > 0:
            prompt_enc = np.hstack([prompt_enc, np.zeros(padding_required, np.int32)])
            
        y = gpt(prompt_enc[None, ...])
        
        idx = -1
        if padding_required >0:
            idx = (-padding_required)-1

        log_logits = nn.log_softmax(y[0][idx])

        # sample next token 
        # TODO: understand multinomial and categorical
        pred_token = jax.random.categorical(gen(), log_logits)
        
        if padding_required >0:
            # remove padding
            prompt_enc = prompt_enc[:-padding_required]
            padding_required = padding_required -1
        prompt_enc = np.hstack([prompt_enc, np.array([pred_token])])
        res = np.hstack([res, np.array([pred_token])])

        
    return tokenizer.decode(res.tolist())
