import paddle
import paddle.nn as nn

class EmbeddingLayer(nn.Layer):
    def __init__(self, emb):
        super(EmbeddingLayer, self).__init__()
        self.emb = emb
        self.weight = emb.weight
    
    def forward(self, x):
        if len(x.shape) == 2:
            y = self.emb(x)
        else:
            y = paddle.matmul(x, self.weight)

        return y