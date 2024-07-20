import torch
import torch.nn as nn 

from collections import OrderedDict

import torchvision.models as tvmodels

from iamcl2r.utils import l2_norm
from iamcl2r.models.resnet import resnet18, resnet50
from iamcl2r.models.senet import SENet18
from iamcl2r.models.regnet import RegNetY_400MF


import logging
logger = logging.getLogger('Model')


__BACKBONE_OUT_DIM = {
    'resnet18': 512,
    'resnet18_torchvision': 512,
    'resnet50_torchvision': 2048,
    'resnet101_torchvision': 2048,
    'senet18': 512,
    'regnet400': 384,
}


def get_backbone_feat_size(backbone):
    if backbone not in __BACKBONE_OUT_DIM:
        raise ValueError('Backbone not supported: {}'.format(backbone))
    return __BACKBONE_OUT_DIM[backbone]


def extract_features(args, device, net, loader, return_labels=False, n_backward_steps=0):
    features = None
    labels = None
    net.eval()
    with torch.no_grad():
        for inputs in loader:
            images = inputs[0].to(device)
            targets = inputs[1]
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                if len(images.size()) == 5:
                    b, n_views, c, h, w = images.size()
                    images = images.view(-1, c, h, w)
                    targets = torch.stack([targets for _ in range(n_views)], dim=1).view(-1)
                f = net.bc_forward(images, n_backward_steps=n_backward_steps)['features']
            f = l2_norm(f)
            if features is not None:
                features = torch.cat((features, f), 0)
                labels = torch.cat((labels, targets), 0) if return_labels else None
            else:
                features = f
                labels = targets if return_labels else None
    if return_labels:
        return features.detach().cpu(), labels.detach().cpu()
    return features.detach().cpu().numpy()


class Incremental_ResNet(nn.Module):
    def __init__(self, 
                 num_classes=100, 
                 feat_size=99, 
                 backbone='resnet18', 
                 pretrained=False,
                ):
        
        super(Incremental_ResNet, self).__init__()
        self.feat_size = feat_size
        
        if backbone == 'resnet18':
            self.backbone = resnet18()
            self.out_dim = self.backbone.out_dim
        elif backbone == 'resnet50':
            self.backbone = resnet50()
            self.out_dim = self.backbone.out_dim
        elif backbone == 'senet18':
            self.backbone = SENet18()
            self.out_dim = self.backbone.out_dim
        elif backbone == 'regnet400':
            self.backbone = RegNetY_400MF()
            self.out_dim = self.backbone.out_dim
        elif backbone == 'resnet18_torchvision':
            self.backbone = tvmodels.resnet18(pretrained=pretrained)
            self.out_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet50_torchvision':
            self.backbone = tvmodels.resnet50(pretrained=pretrained)
            self.out_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet101_torchvision':
            self.backbone = tvmodels.resnet101(pretrained=pretrained)
            self.out_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError('Backbone not supported: {}'.format(backbone))

        self.fc1 = None 
        self.fc2 = None
        if self.out_dim != self.feat_size:
            logger.info(f"add a linear layer from {self.out_dim} to {self.feat_size}")
            self.fc1 = nn.Linear(self.out_dim, self.feat_size, bias=False)
        self.fc2 = nn.Linear(self.feat_size, num_classes, bias=False)  # classifier

            
    def forward(self, x, return_dict=True):
        x = self.backbone(x)

        if self.fc1 is not None:
            z = self.fc1(x)
        else:
            z = x
        
        y = self.fc2(z)

        if return_dict:
            return {'backbone_features': x,
                    'logits': y, 
                    'features': z
                    }

        else:
            return x, y, z
    
    def expand_classifier(self, new_classes):
        old_classes = self.fc2.weight.data.shape[0]
        old_weight = self.fc2.weight.data
        self.fc2 = nn.Linear(self.feat_size, old_classes + new_classes, bias=False)
        self.fc2.weight.data[:old_classes] = old_weight


class BC_Incremental_ResNet(Incremental_ResNet):
    def __init__(self, 
                 num_classes=100, 
                 feat_size=99, 
                 backbone='resnet18', 
                 pretrained=False,
                 n_backward_vers=0,
                 version=0
                ):
        super(BC_Incremental_ResNet, self).__init__(num_classes, feat_size, backbone, pretrained)
        self.n_backward_vers = n_backward_vers
        self.version = version
        self.bc_projs = nn.ModuleList([
            nn.Linear(self.feat_size, self.feat_size, bias=False) for _ in range(n_backward_vers)
        ])

    def bc_forward(self, x, return_dict=True, n_backward_steps=0):
        assert n_backward_steps <= len(self.bc_projs), f"n_backward_steps {n_backward_steps} must be less than or equal to the number of projections {len(self.bc_projs)}"
        x, y, z = self.forward(x, return_dict=False)
        if n_backward_steps > 0:
            for i in range(n_backward_steps):
                z = self.bc_projs[i](z)

        if return_dict:
            return {'backbone_features': x,
                    'logits': y, 
                    'features': z
                    }
        else:
            return x, y, z

    def add_bc_projection(self, require_grad=True):
        new_bc_proj = nn.Linear(self.feat_size, self.feat_size, bias=True)
        if require_grad:
            new_bc_proj.weight.requires_grad = True
        self.bc_projs.insert(0, new_bc_proj)
        

def dsimplex(num_classes=100, device='cuda'):
    def simplex_coordinates_gpu(n, device):
        t = torch.zeros((n + 1, n), device=device)
        torch.eye(n, out=t[:-1,:], device=device)
        val = (1.0 - torch.sqrt(1.0 + torch.tensor([n], device=device))) / n
        t[-1,:].add_(val)
        t.add_(-torch.mean(t, dim=0))
        t.div_(torch.norm(t, p=2, dim=1, keepdim=True)+ 1e-8)
        return t
        
    feat_dim = num_classes - 1
    ds = simplex_coordinates_gpu(feat_dim, device)#.cpu()
    return ds


def create_model(args, 
                 device,
                 resume_path=None, 
                 num_classes=None, 
                 feat_size=None, 
                 backbone=None, 
                 new_classes=None,
                 n_backward_vers=None,
                 **kwargs):
    if backbone is None:
        backbone = args.current_backbone
    elif backbone == 'from_ckpt':
        if resume_path in [None, '', 'torchvision_pretrained']:
            raise ValueError('Backbone not set and no checkpoint provided')
        new_pretrained_dict = torch.load(resume_path, map_location='cpu')
        backbone = new_pretrained_dict['args'].current_backbone
        num_classes = new_pretrained_dict['net']['fc2.weight'].shape[0]

    if feat_size is None and not args.use_embedding_layer:
        if args.fixed:
            feat_size = args.preallocated_classes - 1
        else:
            feat_size = get_backbone_feat_size(backbone)
        args.feat_size = feat_size
    elif args.use_embedding_layer:
        assert args.feat_size is not None, "feat_size must be set in configs when using embedding layer"
        feat_size = args.feat_size
    elif feat_size is None:
        raise ValueError('feat_size not set')

    if num_classes is None:
        num_classes = args.classes_at_task[args.current_task_id-1]

    if new_classes is None and args.current_task_id > 0 and not args.fixed:
        new_classes = len(args.new_data_ids_at_task[args.current_task_id])
        logger.info("new_data_ids_at_task: {}".format(args.new_data_ids_at_task[args.current_task_id]))
    elif args.fixed:
        num_classes = args.preallocated_classes

    logger.info(f"Creating model with {num_classes} classes and {feat_size} features")

    model_cfg = {
        'num_classes': num_classes,
        'feat_size': feat_size,
        'backbone': backbone,
        'pretrained': (resume_path == 'torchvision_pretrained'),
    }
    logger.info(f"Loading a model with config: {model_cfg}")
    if 'bcp' in args.method:
        model_cfg = {
            **model_cfg,
            'n_backward_vers': n_backward_vers,
        }
        model = BC_Incremental_ResNet(**model_cfg)
    else:
        model = Incremental_ResNet(**model_cfg)

    if args.fixed:
        fixed_weights = dsimplex(num_classes=num_classes, device=device)
        logger.info(f"Fixed weights shape: {fixed_weights.shape}")
        model.fc2.weight.requires_grad = False  # set no gradient for the fixed classifier
        model.fc2.weight.copy_(fixed_weights)   # set the weights for the classifier

    if resume_path not in [None, '', 'torchvision_pretrained']:
        logger.info(f"Resuming Weights from {resume_path}")
        new_pretrained_dict = torch.load(resume_path, map_location='cpu')
        if "net" in new_pretrained_dict.keys():
            new_pretrained_dict = new_pretrained_dict["net"]

        if "pretrained" in resume_path:
            state_dict = OrderedDict()
            for k, v in new_pretrained_dict.items():
                name = k.replace('.blocks.', '.')
                if name not in model.state_dict().keys():
                    logger.info(f"{name} \t not found!!!!!!")
                    continue
                state_dict[name] = v
            del state_dict['fc2.weight'] # remove classifier weights from iamcl2r pretrained weights
        else:
            state_dict = new_pretrained_dict

        model.load_state_dict(state_dict, strict=False)    # the dict does not have always the old_classifier weights
    
    if new_classes is not None and new_classes > 0 and not args.fixed:
        logger.info(f"Expanding classifier to {num_classes + new_classes} classes")
        model.expand_classifier(new_classes)
    
    model.to(device=device)
    if args.distributed:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

    return model

