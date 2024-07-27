import os.path as osp
import importlib
from torch.utils.data import DataLoader
from torchvision import transforms
from continuum import ClassIncremental, InstanceIncremental
from continuum.rehearsal import RehearsalMemory

from .scenarios import NoContinualLearning

from iamcl2r.dataset.dataset_utils import *


__factory = {
    'cifar':    'iamcl2r.dataset.cifar.load_cifar',
    'cifar_clr':    'iamcl2r.dataset.cifar_clr.load_cifar_clr',
}


__factory_modalities = ['train', 
                        'identification'
                       ]


def create_data_and_transforms(args, mode="train", return_dict=True):

    if args.train_dataset_name not in __factory.keys():
        raise KeyError(f"Unknown dataset: {args.train_dataset_name}")
    
    if mode not in __factory_modalities:
        raise KeyError(f"Unknown modality: {mode}")
    
    if not osp.exists(args.data_path):
        raise ValueError(f"Data path does not exist: {args.data_path}")
    
    scenario_train = None
    scenario_val = None
    memory = None 
    query_loader = None
    gallery_loader = None
    target_transform = None

    if mode=="train":
        if args.fixed:
            # local classes are written from the bottom of the shared interface w the server
            target_transform = transforms.Lambda(lambda y: args.preallocated_classes - 1 - y)
            
        logger.info(f"Loading Datasets")        
        module_path = '.'.join(__factory[args.train_dataset_name].split('.')[:-1])
        module = importlib.import_module(module_path)
        class_name = __factory[args.train_dataset_name].split('.')[-1]
        
        data_kwargs = {"path": args.data_path,
                       "use_subsampled_dataset": args.use_subsampled_dataset,
                       "img_per_class": args.img_per_class,
                       "input_size": args.input_size,
                      }
        data = getattr(module, class_name)(**data_kwargs)
        args.train_transform = data["train_transform"]

        if args.scenario == 'class-incremental':
            # create task-sets for sequential fine-tuning learning
            assert args.initial_increment is not None and args.increment is not None, "Please provide the initial increment and the increment for the class-incremental scenario."
            scenario_train = ClassIncremental(data["dataset_train"],
                                              initial_increment=args.initial_increment,
                                              increment=args.increment,
                                              transformations=data["train_transform"]
                                              ) 
            args.num_classes = scenario_train.nb_classes
            args.nb_tasks = scenario_train.nb_tasks
            args.nb_tasks_evaluation = scenario_train.nb_tasks 

            logger.info(f"\n\nTraining with {args.nb_tasks} tasks.\nIn the first task there are {args.initial_increment} classes, while the other tasks have {args.increment} classes each.\n\n")

            scenario_val = ClassIncremental(data["dataset_val"],
                                            initial_increment=args.initial_increment,
                                            increment=args.increment,
                                            transformations=data["val_transform"]
                                            ) 
        elif args.scenario == 'instance-incremental':
            assert args.nb_tasks is not None, "Please provide the number of tasks for the instance-incremental scenario."
            scenario_train = InstanceIncremental(data["dataset_train"],
                                                 nb_tasks=args.nb_tasks,
                                                 transformations=data["train_transform"])
            args.num_classes = scenario_train.nb_classes
            args.nb_tasks_evaluation = scenario_train.nb_tasks

            logger.info(f"\n\nTraining with {args.nb_tasks} tasks.")

            scenario_val = InstanceIncremental(data["dataset_val"],
                                               nb_tasks=args.nb_tasks,
                                               transformations=data["val_transform"])
        elif args.scenario == 'none':
            assert args.nb_tasks is not None, "Please provide the number of tasks for none continual learning."
            scenario_train = NoContinualLearning(data["dataset_train"],
                                                 nb_tasks=args.nb_tasks,
                                                 transformations=data["train_transform"])
            args.num_classes = scenario_train.nb_classes
            args.nb_tasks_evaluation = scenario_train.nb_tasks

            scenario_val = NoContinualLearning(data["dataset_val"],
                                               nb_tasks=args.nb_tasks,
                                               transformations=data["val_transform"])
        else:
            raise ValueError(f"Unknown scenario: {args.scenario}")


        
        # create episodic memory dataset
        memory = RehearsalMemory(memory_size=args.num_classes * args.rehearsal,
                                 herding_method="random",
                                 fixed_memory=True,
                                 nb_total_classes=args.num_classes
                                )

        
        args.classes_at_task = []
        args.new_data_ids_at_task = []
        for task_id in range(args.nb_tasks):
            train_task_set = scenario_train[task_id]

            new_data_ids = train_task_set.get_classes()

            class_in_step = (
                scenario_train[:task_id].nb_classes + len(new_data_ids)
                if task_id > 0
                else train_task_set.nb_classes
            )
            args.classes_at_task.append(class_in_step)
            args.new_data_ids_at_task.append(new_data_ids)
        
        if return_dict:
            return {"scenario_train": scenario_train,
                    "scenario_val": scenario_val,
                    "memory": memory, 
                    "target_transform": target_transform
                    }

        return scenario_train, scenario_val, memory, target_transform

    else:        
        module_path = '.'.join(__factory[args.train_dataset_name].split('.')[:-1])
        module = importlib.import_module(module_path)
        class_name = __factory[args.train_dataset_name].split('.')[-1] + f'_{mode}'
        try:
            gallery_set, query_set = getattr(module, class_name)(path=args.data_path)
        except AttributeError:
            raise AttributeError(f"Please implement the function 'load_{args.train_dataset_name}_{mode}'")
        
        query_loader = DataLoader(query_set, batch_size=args.batch_size, 
                                shuffle=False, drop_last=False, 
                                num_workers=args.num_workers)
        gallery_loader = DataLoader(gallery_set, batch_size=args.batch_size,
                                    shuffle=False, drop_last=False, 
                                    num_workers=args.num_workers)
        
        if return_dict:
            return {"query_loader": query_loader, 
                    "gallery_loader": gallery_loader, 
                    }
        return query_loader, gallery_loader, target_transform
