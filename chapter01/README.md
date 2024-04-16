# 大语言模型的基础知识

##  一、用GPT3.5生成的自注意力代码示例

**Question:**  用python写一段自注意力实现代码

**Answer:** 

下面是一个简单的自注意力实现代码示例，使用了PyTorch库：

```python
import torch
import torch.nn.functional as F

class SelfAttention(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.W_q = torch.nn.Linear(input_dim, hidden_dim)
        self.W_k = torch.nn.Linear(input_dim, hidden_dim)
        self.W_v = torch.nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        attention_scores = torch.matmul(q, k.transpose(1, 2)) / $sqrt(hidden_dim)$
        attention_weights = F.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_weights, v)
        return output
```

这里实现了一个简单的自注意力模块，其中`input_dim`是输入特征的维度，`hidden_dim`是注意力计算中的隐藏层维度。在`forward`方法中，首先通过三个线性变换层将输入$x$分别映射为$q$、$k$和$v$，然后计算注意力分数，通过softmax函数得到注意力权重，最后利用注意力权重对$v$进行加权求和得到输出。

## 二、用GPT3.5生成的多头注意力代码示例

**Question:** 用python写一段多头注意力实现代码

### Answer（Tensorflow版本）: 

以下是一个简单的多头注意力实现代码示例，使用Python编写：

```python
import tensorflow as tf

def multihead_attention(Q, K, V, num_heads, head_size):
    # 分别计算Q、K、V的维度
    d_model = Q.shape[-1]
    
    # 将Q、K、V分别线性变换为多头
    Q = tf.keras.layers.Dense(d_model)(Q)
    K = tf.keras.layers.Dense(d_model)(K)
    V = tf.keras.layers.Dense(d_model)(V)
    
    # 将Q、K、V分别拆分为多个头
    Q = tf.reshape(Q, (-1, num_heads, head_size, d_model // num_heads))
    K = tf.reshape(K, (-1, num_heads, head_size, d_model // num_heads))
    V = tf.reshape(V, (-1, num_heads, head_size, d_model // num_heads))
    
    # 计算注意力分数
    attention_scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(d_model // num_heads, tf.float32))
    attention_weights = tf.nn.softmax(attention_scores, axis=-1)
    
    # 注意力加权求和
    attention_output = tf.matmul(attention_weights, V)
    attention_output = tf.reshape(attention_output, (-1, head_size, d_model))
    
    return attention_output

# 示例用法
Q = tf.random.normal((32, 10, 64))  # (batch_size, sequence_length, d_model)
K = tf.random.normal((32, 10, 64))  # (batch_size, sequence_length, d_model)
V = tf.random.normal((32, 10, 64))  # (batch_size, sequence_length, d_model)
num_heads = 4
head_size = 16

output = multihead_attention(Q, K, V, num_heads, head_size)
print(output.shape)
```

这段代码实现了一个简单的多头注意力机制，可以根据需要调整输入数据的维度和参数设置。

### Answer（PyTorch版本）: 

以下是一个使用PyTorch编写的多头注意力实现代码示例：

```python
import torch
import torch.nn.functional as F

class MultiheadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_size = d_model // num_heads
        
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V):
        batch_size = Q.size(0)
        
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.head_size)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.head_size)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.head_size)
        
        attention_scores = torch.matmul(Q, K.transpose(2, 3)) / (self.head_size ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.view(batch_size, -1, self.d_model)
        
        return attention_output

# 示例用法
Q = torch.randn(32, 10, 64)  # (batch_size, sequence_length, d_model)
K = torch.randn(32, 10, 64)  # (batch_size, sequence_length, d_model)
V = torch.randn(32, 10, 64)  # (batch_size, sequence_length, d_model)
num_heads = 4

multihead_attention = MultiheadAttention(d_model=64, num_heads=num_heads)
output = multihead_attention(Q, K, V)
print(output.shape)
```

这段代码实现了一个简单的多头注意力机制，使用PyTorch框架。可以根据需要调整输入数据的维度和参数设置。