import torch
import torch.nn as nn
from torchvision import models
from mains.embryo_pretrain._resnet import resnet50

class MultiTaskHead(nn.Module):
    def __init__(self, n_classes, in_size=4096):
        super(MultiTaskHead, self).__init__()
        self.fcs = nn.ModuleList([nn.Sequential(
            nn.Linear(in_size, in_size),
            nn.ReLU(),
            nn.Linear(in_size, n_class)
        ) for n_class in n_classes])

    def forward(self, x):
        return [fc(x) for fc in self.fcs]


class MultiTaskLoss(nn.Module):
    def __init__(self, n_classes):
        super(MultiTaskLoss, self).__init__()
        self.n_classes = n_classes
        self.loss_fns = []
        for n_class in self.n_classes:
            if n_class != 1:
                self.loss_fns.append(nn.CrossEntropyLoss(reduction='none'))
            else:
                self.loss_fns.append(nn.MSELoss(reduction='none'))

    def forward(self, preds, labels):
        for i in range(len(preds)):
            if self.n_classes[i] != 1:
                labels[i] = labels[i].long()
        sub_loss = [self.loss_fns[i](preds[i], labels[i]) for i in range(len(preds))]
        loss = torch.cat(sub_loss)
        loss = torch.sum(loss, dim=0)
        sub_loss_sum = [torch.sum(sub_loss[i], dim=-1) for i in range(len(sub_loss))]
        sub_loss_mean = [torch.mean(sub_loss[i], dim=-1) for i in range(len(sub_loss))]
        return loss, [sub_loss_sum, sub_loss_mean]


class EmbryoImageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predict = False
        self.config = config
        self.arch = self.config['arch']
        self.n_classes = self.config['n_classes']
        self.loss_fn = MultiTaskLoss(self.n_classes)

        if self.arch == 'resnet18':
            self.net = models.resnet18(pretrained=True)
            n_out = 512
        elif self.arch == 'resnet50':
            if self.config['multimodal']:
                self.net = resnet50(pretrained=False)
            else:
                self.net = models.resnet50(pretrained=True)
            n_out = 2048
        # morphology
        self.net.fc = MultiTaskHead(self.n_classes, in_size=n_out)
        # Multimodal
        # self.net.fc = MultiTaskHead(self.n_classes, in_size=n_out + 256)
        # MLP
        # self.net.fc = MultiTaskHead(self.n_classes, in_size=256)

    
    def backbone_parameters(self):
        return map(lambda kv: kv[1], filter(lambda kv: not kv[0].startswith('fc.'), self.net.named_parameters()))

    def head_parameters(self):
        return self.net.fc.parameters()

    def _forward(self, img):
        return self.net(img)

    # # begin clinical_factor
    # def _multimodal_forward(self, img, factor):
    #     return self.net(img, factor)

    # def forward(self, img, factor, label=None):
    #     pred = self._multimodal_forward(img, factor)
    #     if self.predict:
    #         for i in range(len(pred)):
    #             if self.n_classes[i] != 1:
    #                 pred[i] = torch.softmax(pred[i], dim=-1)
    #         return pred
    #     else:
    #         for i in range(len(pred)):
    #             mask = ~torch.isnan(label[i])
    #             if self.n_classes[i] == 1:
    #                 pred[i] = pred[i].squeeze(-1)
    #                 pred[i] = pred[i][mask]
    #             else:
    #                 pred[i] = pred[i][mask, :]
    #             label[i] = label[i][mask]
    #         loss = None
    #         if label is not None:
    #             # loss = self.loss_fn(pred, label).unsqueeze(0)
    #             loss, _ = self.loss_fn(pred, label)
    #         return pred, loss.unsqueeze(0)
    # # end clinical_factor

    def forward(self, img, label=None):
        pred = self._forward(img)       
        if self.predict:
            for i in range(len(pred)):
                if self.n_classes[i] != 1:
                    pred[i] = torch.softmax(pred[i], dim=-1)
            return pred
        else:
            for i in range(len(pred)):
                mask = ~torch.isnan(label[i])
                if self.n_classes[i] == 1:
                    pred[i] = pred[i].squeeze(-1)
                    pred[i] = pred[i][mask]
                else:
                    pred[i] = pred[i][mask, :]
                label[i] = label[i][mask]
            loss = None
            if label is not None:
                loss, sub_loss = self.loss_fn(pred, label)
            return pred, loss.unsqueeze(0), sub_loss