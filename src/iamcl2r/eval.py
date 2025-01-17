import numpy as np
import os.path as osp
import wandb

from iamcl2r.models import create_model, get_backbone_feat_size, extract_features
from iamcl2r.compatibility_metrics import average_compatibility, average_accuracy
from iamcl2r.performance_metrics import identification
from iamcl2r.visualize import visualize_compatibility_matrix

import logging
logger = logging.getLogger('Eval')


def evaluate(args, device, query_loader, gallery_loader, ntasks_eval=None, is_post_hoc=False, topk=1):
    if ntasks_eval is None: 
        ntasks_eval = args.ntasks_eval
    compatibility_matrix = np.zeros((ntasks_eval, ntasks_eval))
    targets = query_loader.dataset.targets
    gallery_targets = gallery_loader.dataset.targets

    for task_id in range(ntasks_eval):
        ckpt_name = f"ckpt_{task_id}_aligned.pt" if is_post_hoc else f"ckpt_{task_id}.pt"
        ckpt_path = osp.join(*(args.checkpoint_path, ckpt_name))
        if not osp.exists(ckpt_path):
            if is_post_hoc and not osp.exists(ckpt_path.replace("_aligned", "")):
                raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist. All the checkpoints need to have the format 'ckpt_<id>.pt' where id is the task id.")
            else:
                ckpt_path = ckpt_path.replace("_aligned", "")
        net = create_model(args,
                           device,
                           resume_path=ckpt_path, 
                           num_classes='from_ckpt', 
                           backbone='from_ckpt',
                           new_classes=0,
                           feat_size=args.feat_size,
                           n_backward_vers=task_id + 1,
                          )
        net.eval() 

        for i in range(task_id+1):
            ckpt_name = f"ckpt_{i}_aligned.pt" if is_post_hoc else f"ckpt_{i}.pt"
            ckpt_path = osp.join(*(args.checkpoint_path, ckpt_name))
            if not osp.exists(ckpt_path):
                if is_post_hoc and not osp.exists(ckpt_path.replace("_aligned", "")):
                    raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist. All the checkpoints need to have the format 'ckpt_<id>.pt' where id is the task id.")
                else:
                    ckpt_path = ckpt_path.replace("_aligned", "")
                # raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist. All the checkpoints need to have the format 'ckpt_<id>.pt' where id is the task id.")
            previous_net = create_model(args,
                                        device,
                                        resume_path=ckpt_path, 
                                        num_classes='from_ckpt',
                                        backbone='from_ckpt',
                                        new_classes=0,
                                        feat_size=args.feat_size,
                                        n_backward_vers=task_id,
                                        )
            previous_net.eval() 
            
            query_feat = extract_features(args, device, net, query_loader, n_backward_steps=task_id-i)
            gallery_feat = extract_features(args, device, previous_net, gallery_loader)

            acc = identification(gallery_feat, gallery_targets, 
                                 query_feat, targets, 
                                 topk=topk
                                )

            compatibility_matrix[task_id][i] = acc
            if i != task_id:
                acc_str = f'Cross-test accuracy between model at task {task_id+1} and {i+1}:'
            else:
                acc_str = f'Self-test of model at task {i+1}:'
            acc_str += f' 1:N search acc: {acc:.2f}'
            logger.info(f'{acc_str}')
        
    logger.info(f"Compatibility Matrix:\n{compatibility_matrix}")

    if compatibility_matrix.shape[0] > 1:
        # compatibility metrics
        ac = average_compatibility(matrix=compatibility_matrix)
        am = average_accuracy(matrix=compatibility_matrix)

        logger.info(f"Avg. Comp. = {ac:.2f}")
        logger.info(f"AM. Comp. = {am:.3f}")

        if args.is_main_process:
            wandb.log({f"eval/comp-acc": ac, 
                    f"eval/comp-am": am,
                    })

        # create a txt file with the compatibility matrix printed
        with open(osp.join(*(args.checkpoint_path, f'comp-matrix.txt')), 'w') as f:
            f.write(f"Compatibility Matrix ID:\n{compatibility_matrix}\n")
            f.write(f"Avg. Comp. = {ac:.2f}\n")
            f.write(f"AM. Comp. = {am:.3f}\n")


    file_path = osp.join(*(args.checkpoint_path, f'comp-matrix.png'))
    visualize_compatibility_matrix(compatibility_matrix, file_path)
    return compatibility_matrix


def validation(args, device, net, query_loader, gallery_loader, task_id, selftest=False):
    targets = query_loader.dataset.targets
    gallery_targets = gallery_loader.dataset.targets
        
    net.eval() 
    query_feat = extract_features(args, device, net, query_loader)
    
    if selftest:
        previous_net = net
    else:
        ckpt_path_val = osp.join(*(args.checkpoint_path, f"ckpt_{task_id-1}.pt")) 
        if args.fixed and not args.maximum_class_separation: 
            num_classes = args.preallocated_classes
        else:
            num_classes = args.classes_at_task[task_id-1]
        if args.replace_model_architecture:
            raise NotImplementedError("Change model arch not implemented in evaluation")
        else:
            backbone = args.backbone
        logger.info(f"backbone: {backbone}")
        previous_net = create_model(args,
                                    device,
                                    resume_path=ckpt_path_val, 
                                    num_classes=num_classes, 
                                    backbone=backbone,
                                    feat_size=args.feat_size,
                                    )
        previous_net.eval() 
        previous_net.to(device)
    
    gallery_feat = extract_features(args, device, previous_net, gallery_loader)
    acc = identification(gallery_feat, gallery_targets, 
                         query_feat, targets, 
                         topk=1)
    logger.info(f"{'Self' if selftest else 'Cross'} 1:N search Accuracy: {acc*100:.2f}")
    return acc


