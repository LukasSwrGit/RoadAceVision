import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
#from efficientvit.models.efficientvit.cls import efficientvit_cls_b1


### Model Groups Below ###

### ResNet ###

class ResNet18Classifier(nn.Module):
    def __init__(self, nr_classes: int) -> None:
        super(ResNet18Classifier, self).__init__()

        # Load pretrained ResNet50
        self.base_model = models.resnet18(pretrained=True)

        # Freeze the feature extractor 
        for param in self.base_model.parameters():
             param.requires_grad = True

       
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, nr_classes)

    def forward(self, x):
        return self.base_model(x)   
class ResNet34Classifier(nn.Module):
    def __init__(self, nr_classes: int) -> None:
        super(ResNet34Classifier, self).__init__()

        # Load pretrained ResNet50
        self.base_model = models.resnet34(pretrained=True)

        # Freeze the feature extractor 
        for param in self.base_model.parameters():
             param.requires_grad = True

       
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, nr_classes)

    def forward(self, x):
        return self.base_model(x)
class ResNet50Classifier(nn.Module):
    def __init__(self, nr_classes: int) -> None:
        super(ResNet50Classifier, self).__init__()

        # Load pretrained ResNet50
        self.base_model = models.resnet50(pretrained=True)

        # Freeze the feature extractor 
        for param in self.base_model.parameters():
             param.requires_grad = True

       
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, nr_classes)

    def forward(self, x):
        return self.base_model(x)

### ConvNeXT ###

class ConvNeXtTinyClassifier(nn.Module):
    def __init__(self, nr_classes: int) -> None:
        super(ConvNeXtTinyClassifier, self).__init__()

        # Load pretrained ConvNeXt-Tiny model
        self.base_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)

        # Freeze the feature extractor
        for param in self.base_model.parameters():
            param.requires_grad = True

        # Replace the classifier head
        in_features = self.base_model.classifier[2].in_features
        self.base_model.classifier[2] = nn.Linear(in_features, nr_classes)

    def forward(self, x):
        return self.base_model(x) 
class ConvNeXtSmallClassifier(nn.Module):
    def __init__(self, nr_classes: int) -> None:
        super(ConvNeXtSmallClassifier, self).__init__()

        # Load pretrained ConvNeXt-Tiny model
        self.base_model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)

        # Freeze the feature extractor
        for param in self.base_model.parameters():
            param.requires_grad = True

        # Replace the classifier head
        in_features = self.base_model.classifier[2].in_features
        self.base_model.classifier[2] = nn.Linear(in_features, nr_classes)

    def forward(self, x):
        return self.base_model(x)
class ConvNeXtBaseClassifier(nn.Module):
    def __init__(self, nr_classes: int) -> None:
        super(ConvNeXtBaseClassifier, self).__init__()

        # Load pretrained ConvNeXt-Tiny model
        self.base_model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)

        # Freeze the feature extractor
        for param in self.base_model.parameters():
            param.requires_grad = True

        # Replace the classifier head
        in_features = self.base_model.classifier[2].in_features
        self.base_model.classifier[2] = nn.Linear(in_features, nr_classes)

    def forward(self, x):
        return self.base_model(x)

### VisionTransformer ###

class ViT_B_32_Classifier(nn.Module):
    def __init__(self, nr_classes: int) -> None:
        super(ViT_B_32_Classifier, self).__init__()
        self.base_model = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)

        for param in self.base_model.parameters():
            param.requires_grad = True

        in_features = self.base_model.heads.head.in_features
        self.base_model.heads.head = nn.Linear(in_features, nr_classes)

    def forward(self, x):
        return self.base_model(x)  
class ViT_B_16_Classifier(nn.Module):
    def __init__(self, nr_classes: int) -> None:
        super(ViT_B_16_Classifier, self).__init__()
        self.base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

        for param in self.base_model.parameters():
            param.requires_grad = True

        in_features = self.base_model.heads.head.in_features
        self.base_model.heads.head = nn.Linear(in_features, nr_classes)

    def forward(self, x):
        return self.base_model(x)
class ViT_L_32_Classifier(nn.Module):
    def __init__(self, nr_classes: int) -> None:
        super(ViT_L_32_Classifier, self).__init__()
        self.base_model = models.vit_l_32(weights=models.ViT_L_32_Weights.DEFAULT)

        for param in self.base_model.parameters():
            param.requires_grad = True

        in_features = self.base_model.heads.head.in_features
        self.base_model.heads.head = nn.Linear(in_features, nr_classes)

    def forward(self, x):
        return self.base_model(x)
class ViT_L_16_Classifier(nn.Module):
    def __init__(self, nr_classes: int) -> None:
        super(ViT_L_16_Classifier, self).__init__()
        self.base_model = models.vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT)

        for param in self.base_model.parameters():
            param.requires_grad = True

        in_features = self.base_model.heads.head.in_features
        self.base_model.heads.head = nn.Linear(in_features, nr_classes)

    def forward(self, x):
        return self.base_model(x)

### EfficientVIT ###

class EfficientViTClassifier(nn.Module):
    def __init__(self, nr_classes: int) -> None:
        super().__init__()
        self.base_model = efficientvit_cls_b1(n_classes=nr_classes)

        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.base_model(x)
    
### Fusion Architectures ###

class MMTM(nn.Module):
    ##MMTM Notes:
    #The input vectors are concatenated, compressed for high density information, and then expanded to the size of vector 1 after gate 1 and vector 2 after gate 2.
    #The sigmoid activation function is used to ensure the scaling factors are between 0 and 1
    #The compression step mixes the information from both input vectors and disrupts sequential form (50% v1 and 50% v2 get mixed so not sequential anymore)
    #Finally this factor containing the cross attention infromation is multiplied with the original feature vectors to get the recalibrated output vectors
    def __init__(self, ch1, ch2, reduction=16):
        super().__init__()
        self.reduce = reduction
        self.gate1 = nn.Sequential(
            nn.Linear(ch1 + ch2, (ch1 + ch2) // self.reduce),
            nn.ReLU(),
            nn.Linear((ch1 + ch2) // self.reduce, ch1),
            nn.Sigmoid()
        )
        self.gate2 = nn.Sequential(
            nn.Linear(ch1 + ch2, (ch1 + ch2) // self.reduce),
            nn.ReLU(),
            nn.Linear((ch1 + ch2) // self.reduce, ch2),
            nn.Sigmoid()
        )

    def forward(self, f1, f2):  # f1, f2: [B, C]
        z = torch.cat([f1, f2], dim=1)  # [B, C+C]
        s1 = self.gate1(z)
        s2 = self.gate2(z)
        return f1 * s1, f2 * s2

class FlexibleFusionClassifier(nn.Module):
    def __init__(self, backbones: list[nn.Module], feature_dims: list[int], nr_classes=3):
        super().__init__()
        self.backbones = nn.ModuleList(backbones)
        self.feature_dims = feature_dims
        self.num_modalities = len(backbones)

        # Create pairwise MMTMs for adjacent modalities
        self.mmtms = nn.ModuleList([
            MMTM(d1, d2) for d1, d2 in zip(feature_dims[:-1], feature_dims[1:])
        ])

        self.classifier = nn.Sequential(
            nn.Linear(sum(feature_dims), 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, nr_classes)
        )

    def forward(self, *inputs):
        assert len(inputs) == self.num_modalities
        with torch.no_grad():
            features = []
            for b, x in zip(self.backbones, inputs):
                f = b(x)
                if f.ndim == 4:
                    f = torch.nn.functional.adaptive_avg_pool2d(f, 1).view(f.size(0), -1)
                elif f.ndim == 3:
                    f = f.mean(dim=1)
                features.append(F.normalize(f, p=2, dim=1))

        # Apply MMTM fusion
        fused = [features[0]]
        for i in range(1, self.num_modalities):
            f1, f2 = self.mmtms[i - 1](fused[-1], features[i])
            fused[-1] = f1
            fused.append(f2)

        concat = torch.cat(fused, dim=1)
        return self.classifier(concat)

def build_backbones(configs: list[dict], device):
    backbones = []
    feature_dims = []

    for config in configs:
        model_class = config["network"]
        model_path = config["path"]
        nr_classes = config.get("nr_classes", 3)

        model = model_class(nr_classes=nr_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))

        # Remove classifier head
        if hasattr(model, "base_model"):
            if hasattr(model.base_model, "fc"):
                model.base_model.fc = nn.Identity()
            elif hasattr(model.base_model, "classifier"):
                if isinstance(model.base_model.classifier, nn.Sequential):
                    model.base_model.classifier = nn.Identity()
                elif isinstance(model.base_model.classifier, nn.ModuleList):
                    model.base_model.classifier = nn.Identity()
                elif hasattr(model.base_model.classifier, "head"):
                    model.base_model.classifier.head = nn.Identity()
            elif hasattr(model.base_model, "heads"):
                model.base_model.heads.head = nn.Identity()

        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model = model.to(device)
        backbones.append(model)

        # Infer feature dim
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            out = model(dummy_input)
            if out.ndim == 4:
                out = torch.nn.functional.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
            elif out.ndim == 3:  # e.g. ViT [B, Tokens, C]
                out = out.mean(dim=1)
            feature_dims.append(out.size(1))

    return backbones, feature_dims

