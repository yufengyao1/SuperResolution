import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg19,VGG19_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg_model = vgg19(weights=VGG19_Weights.DEFAULT).features[:].to(device)
vgg_model.eval()
for param in vgg_model.parameters():
    param.requires_grad = False  
    
    
class PerpetualLoss(torch.nn.Module):
    def __init__(self, vgg_model):
        super(PerpetualLoss, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '13': "relu3_2",
            '22': "relu4_2",
            '31': "relu5_2"
        }
        # self.weight = [1/2.6,1/4.8,1/3.7,1/5.6,10/1.5]
        self.weight = [1.0, 1.0, 1.0, 1.0, 1.0]

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, output, gt):
        loss = []
        output_features = self.output_features(output)
        gt_features = self.output_features(gt)
        for iter, (dehaze_feature, gt_feature, loss_weight) in enumerate(zip(output_features, gt_features, self.weight)):
            loss.append(F.mse_loss(dehaze_feature, gt_feature)*loss_weight)
        return sum(loss)

class FFTLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self,y_pred,y_true):
        y_pred_fft = torch.fft.fftn(y_pred)
        y_true_fft = torch.fft.fftn(y_true)
        # 计算两个信号在频域上的欧几里德距离
        l2_distance = torch.norm(y_pred_fft - y_true_fft, p=2)
        return l2_distance
    
    
class HSVLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(HSVLoss, self).__init__()
        self.eps = eps

    def rgb_to_hsv(self, img):

        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

        hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,2]==img.max(1)[0] ]
        hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,1]==img.max(1)[0] ]
        hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,0]==img.max(1)[0] ]) % 6

        hue[img.min(1)[0]==img.max(1)[0]] = 0.0
        hue = hue/6

        saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + self.eps )
        saturation[ img.max(1)[0]==0 ] = 0

        value = img.max(1)[0]
        
        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value],dim=1)
        return hsv
    def forward(self,y_pred,y_true):
        pred=self.rgb_to_hsv(y_pred)
        true=self.rgb_to_hsv(y_true)
        loss=pred-true
        return loss
        


class SRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.loss_l1=nn.L1Loss()
        self.loss_perpetual=PerpetualLoss(vgg_model)
        self.loss_fft=FFTLoss()
        self.loss_hsv=HSVLoss()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to(device)

    def forward(self, label, out):
        # loss_l1 = self.loss_l1(label, out)
        loss_fft=0.00001*self.loss_fft(label,out)
       
        # label = label[0].permute(1, 0, 2, 3)
        # out = out[0].permute(1, 0, 2, 3)
        label = (label - self.mean) / self.std
        out = (out - self.mean) / self.std

        loss_feature = self.loss_perpetual(label, out)
        return loss_feature+loss_fft#+loss_hsv
