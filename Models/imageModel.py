import torch
import torch.nn as nn
import torchvision.models as models

class ImageExpertsModel(nn.Module):
    def __init__(self, cfg):
        super(ImageExpertsModel, self).__init__()
        
        #ResNet50
        #Analyzes standard pixels to find objects, textures, and real-world context
        resnet = models.resnet50(pretrained=True)
        #Remove the final classification layer so we get the raw 2048-dimensional feature vector
        self.spatial_expert = nn.Sequential(*list(resnet.children())[:-1])
        spatial_dim = 2048
        
        #VGG19
        #Analyzes the DFT spectrum to detect deepfake noise, recompression, and artifacts
        vgg = models.vgg19(pretrained=True)
        self.freq_expert = nn.Sequential(
            vgg.features,
            vgg.avgpool,
            nn.Flatten()
        )
        freq_dim = 512 * 7 * 7 #Standard output size of VGG19 features
        
        #Projection layers to align dimensions
        emb_dim = cfg.get("emb_dim", 256)
        self.spatial_proj = nn.Sequential(nn.Linear(spatial_dim, emb_dim), nn.ReLU())
        self.freq_proj = nn.Sequential(nn.Linear(freq_dim, emb_dim), nn.ReLU())
        
        #Gating / Mask Attention Mechanism
        #Decides which expert (Spatial vs. Frequency) to trust more for this specific image
        self.attention = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 2),
            nn.Softmax(dim=1)
        )
        
        #Final Classifier
        mlp_cfg = cfg.get("model", {}).get("mlp", {"dims": [256, 128], "dropout": 0.3})
        mlp_dims = mlp_cfg.get("dims", [256, 128])
        dropout = mlp_cfg.get("dropout", 0.3)
        
        layers = []
        in_dim = emb_dim
        for dim in mlp_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = dim
            
        layers.append(nn.Linear(in_dim, 2))  #output classes: Real (0) or Fake (1)
        self.classifier = nn.Sequential(*layers)

    def forward(self, img_spatial, img_frequency):
        #Extract Features
        s_feat = self.spatial_expert(img_spatial)
        s_feat = torch.flatten(s_feat, 1)
        s_emb = self.spatial_proj(s_feat)
        
        f_feat = self.freq_expert(img_frequency)
        f_emb = self.freq_proj(f_feat)
        
        #Calculate Attention Weights
        concat_emb = torch.cat([s_emb, f_emb], dim=1)
        weights = self.attention(concat_emb) #Output shape: (Batch_Size, 2)
        
        #Apply weights to the embeddings
        s_weighted = s_emb * weights[:, 0].unsqueeze(1)
        f_weighted = f_emb * weights[:, 1].unsqueeze(1)
        
        #Combine and Classify
        fused_emb = s_weighted + f_weighted
        logits = self.classifier(fused_emb)
        
        return logits