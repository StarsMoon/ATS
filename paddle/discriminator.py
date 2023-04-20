from paddle import nn


class Discriminator(nn.Layer):
    def __init__(self, d_model=768):
        super(Discriminator, self).__init__()
        self.classifier = nn.Linear(d_model, 3)
    
    def forward(self, logits):
        logits = self.classifier(logits)
        return logits