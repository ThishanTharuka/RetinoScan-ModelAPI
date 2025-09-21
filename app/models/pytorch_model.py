import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import timm

def gem(x, p=3, eps=1e-6):
    """Generalized Mean (GeM) pooling"""
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    """Generalized Mean pooling layer"""
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
    
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"

class RetinalModel(nn.Module):
    """
    Retinal analysis model using EfficientNet backbone with GeM pooling
    This matches the architecture from your training code
    """
    def __init__(self, model_name='efficientnet_b3', num_classes=1, pretrained=True):
        super(RetinalModel, self).__init__()
        
        # Create backbone model
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        # Get the number of features from the classifier
        in_features = self.backbone.classifier.in_features
        
        # Replace classifier with identity to get features
        self.backbone.classifier = nn.Identity()
        
        # Add custom head
        self.global_pool = GeM()
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        """Forward pass through the model"""
        # Extract features using backbone
        features = self.backbone.forward_features(x)
        
        # Apply GeM pooling
        pooled = self.global_pool(features)
        
        # Flatten for classifier
        pooled = pooled.view(pooled.size(0), -1)
        
        # Apply dropout and final classification
        pooled = self.dropout(pooled)
        output = self.classifier(pooled)
        
        return output

def predict_classes(predictions, thresholds=[0.5, 1.5, 2.5, 3.5]):
    """
    Convert regression predictions to classification classes
    Based on your training code's prediction logic
    """
    predictions = torch.tensor(predictions) if not isinstance(predictions, torch.Tensor) else predictions
    predictions = predictions.cpu().numpy()
    
    classes = torch.zeros_like(torch.tensor(predictions), dtype=torch.int)
    for i, threshold in enumerate(thresholds):
        classes[predictions >= threshold] = i + 1
    
    return classes.numpy().astype(int)