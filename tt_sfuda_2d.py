
import os
import yaml
import argparse
import albumentations as A

from glob import glob
from tqdm import tqdm
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.jit
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as st_transforms

import losses
import archs

cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=None, help='model name')
    parser.add_argument('--target', default=None, help='model name')
    parser.add_argument('--max-train-steps', default=0, type=int,
                        help='Limit adaptation steps per epoch for quick experiments (0 = full epoch).')
    parser.add_argument('--pseudo-thresh', default=0.65, type=float,
                        help='Confidence threshold for pseudo-labels (default 0.65, higher = fewer but cleaner labels).')
    parser.add_argument('--adapt-lr-scale', default=0.2, type=float,
                        help='Scale factor applied to config lr for adaptation optimizer (default 0.2 = lr*0.2).')
    parser.add_argument('--stage1', default=0, type=int,
                        help='Override stage1 epochs (0 = use model config default).')
    parser.add_argument('--stage2', default=0, type=int,
                        help='Override stage2 epochs (0 = use model config default).')
    parser.add_argument('--const-loss-weight', default=0.01, type=float,
                        help='Weight for consistency loss in Stage II (default 0.01). '
                             'Lower values prevent catastrophic forgetting with large encoders.')
    args = parser.parse_args()
    return args

def build_strong_augmentation(img):
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    augmentation = []
    augmentation.append(st_transforms.RandomApply([st_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.6))
    augmentation.append(st_transforms.RandomGrayscale(p=0.2))
    strong_aug = st_transforms.Compose(augmentation)
    s_input = strong_aug(img)
    return s_input

def build_pseduo_augmentation(img):
    aug1 = st_transforms.ColorJitter(0.01, 0.01, 0.01, 0.01)
    aug2 = st_transforms.RandomGrayscale(p=1.0)
    aug3 = st_transforms.RandomSolarize(threshold=0.5, p=1.0)
    aug4 = st_transforms.RandomAutocontrast(p=1.0)

    aug_img1 = aug1(img).unsqueeze(0)
    aug_img2 = aug2(img).unsqueeze(0)
    aug_img3 = aug3(img).unsqueeze(0)
    aug_img4 = aug4(img).unsqueeze(0)
    aug_data = torch.cat([img.unsqueeze(0), aug_img1, aug_img2, aug_img3, aug_img4], dim=0)
    return aug_data

@torch.no_grad()
def update_teacher_model(model_student, model_teacher, keep_rate=0.996):
    student_model_dict = model_student.state_dict()

    new_teacher_dict = OrderedDict()
    for key, value in model_teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                student_model_dict[key] *
                (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))
    return new_teacher_dict

def consistency_loss(msrc_feat, tgt_feat):
    req_feat = [0,1,2,3]
    total_loss = 0 
    loss = nn.MSELoss()
    for i in req_feat:
        total_loss = total_loss + loss(tgt_feat[i], msrc_feat[i])
    return total_loss/len(req_feat)


@torch.jit.script
def sigmoid_entropy_loss(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x*torch.log(x + 1e-30) + (1-x)*torch.log(1-x + 1e-30)).mean()

@torch.jit.script
def sigmoid_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x*torch.log(x + 1e-30) + (1-x)*torch.log(1-x + 1e-30))


def ent_select(aug_all_ent):
    aug_req_ent = []
    for i in range(len(aug_all_ent)): 
        if (aug_all_ent[i]).mean().item() > 0.0001: 
            aug_req_ent.append(aug_all_ent[i])
    return aug_req_ent

def uncert_voting(aug_output, pseudo_thresh=0.65):
    aug_all_prob = []
    aug_all_ent = []
    for i in range(1, len(aug_output)):
        prob = torch.sigmoid(aug_output[i])
        aug_all_prob.append(prob)
        aug_all_ent.append(sigmoid_entropy(prob))
    
    no_aug_prob_nor = torch.sigmoid(aug_output[0])
    no_aug_pseudo_label = no_aug_prob_nor.clone()
    no_aug_pseudo_label[no_aug_pseudo_label>=pseudo_thresh]=1
    no_aug_pseudo_label[no_aug_pseudo_label<pseudo_thresh]=0

    no_aug_ent = sigmoid_entropy(torch.sigmoid(aug_output[0]))
    no_aug_ent[torch.isnan(no_aug_ent)] = 0

    aug_prob_nor = sum(aug_all_prob)/len(aug_all_prob)
    aug_pseudo_label = aug_prob_nor
    aug_pseudo_label[aug_pseudo_label>=pseudo_thresh]=1
    aug_pseudo_label[aug_pseudo_label<pseudo_thresh]=0

    aug_req_ent = ent_select(aug_all_ent)
    aug_avg_ent = sum(aug_req_ent)/len(aug_req_ent)
    aug_avg_ent[torch.isnan(aug_avg_ent)] = 0

    no_aug_ent_nor = ((no_aug_ent - no_aug_ent.min()) * (1/(no_aug_ent.max() - no_aug_ent.min())))
    aug_avg_ent_nor = ((aug_avg_ent - aug_avg_ent.min()) * (1/(aug_avg_ent.max() - aug_avg_ent.min())))

    ent_weight = 0.75
    weighted_ent = ent_weight*no_aug_ent_nor+(1-ent_weight)*aug_avg_ent_nor
    weighted_ent_thresh = weighted_ent.clone()
    weighted_ent_thresh[weighted_ent_thresh>0.5]=1
    weighted_ent_thresh[weighted_ent_thresh<=0.5]=0

    prob_min = 0.3 
    unct_no_aug_prob = no_aug_prob_nor.clone()
    unct_no_aug_prob[unct_no_aug_prob>0.5]=0
    unct_no_aug_prob[unct_no_aug_prob<=prob_min]=0
    unct_no_aug_prob[unct_no_aug_prob>0]=1

    pseudo_uncert = unct_no_aug_prob.int()&weighted_ent_thresh.int()
    pseudo_label = no_aug_pseudo_label.int()|pseudo_uncert.int()
    return pseudo_label.unsqueeze(0).float()

def sfuda_target(config, train_loader, pseduo_model, msrc_model, criterion, optimizer, max_train_steps=0, pseudo_thresh=0.65):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}
    pseduo_model.eval()
    msrc_model.train()
    pbar = tqdm(total=len(train_loader))

    for step_idx, (input, target, path) in enumerate(train_loader, start=1):
        aug_input = build_pseduo_augmentation(input.squeeze(0))
        with torch.no_grad():
            aug_output = pseduo_model(aug_input.to(device))
            ps_output = uncert_voting(aug_output.detach(), pseudo_thresh=pseudo_thresh)

        optimizer.zero_grad()
        output = msrc_model(aug_input.to(device))
        seg_loss = criterion(output.to(device), ps_output.repeat(5,1,1,1).to(device))
        ent_loss = sigmoid_entropy_loss(torch.sigmoid(output))
        loss = seg_loss + ent_loss 
        loss.backward()
        optimizer.step()

        iou,dice = iou_score(output, target)
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
        if max_train_steps > 0 and step_idx >= max_train_steps:
            break
    pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])

def sfuda_task(train_loader, msrc_model, tgt_model, criterion, optimizer, max_train_steps=0, const_loss_weight=0.01):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    msrc_model.eval()
    tgt_model.train()
    pbar = tqdm(total=len(train_loader))

    for step_idx, (input, target, _) in enumerate(train_loader, start=1):
        w_input = input.to(device)
        target = target.to(device)
        image_strong_aug = build_strong_augmentation(input.squeeze(0))
        s_input = image_strong_aug.unsqueeze(0).to(device)

        with torch.no_grad():
            w_output, msrc_feat = msrc_model(w_input, mode='const')
            ps_output = torch.sigmoid(w_output).detach().clone()
            ps_output[ps_output>=0.65]=1
            ps_output[ps_output<0.35]=0
            # Uncertain region [0.35, 0.65) → ignore by setting to -1 is not supported in BCE;
            # keep background for uncertain pixels (safer than forcing foreground)
            ps_output[ps_output!=1]=0

        optimizer.zero_grad()
        output, tgt_feat = tgt_model(s_input, mode='const')
        seg_loss = criterion(output, ps_output)
        const_loss = consistency_loss(msrc_feat, tgt_feat)
        loss = seg_loss + const_loss_weight * const_loss
        loss.backward()
        optimizer.step()

        iou, dice = iou_score(output, target)
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ])
        pbar.set_postfix(postfix)
        pbar.update(1)

        new_msrc_dict = update_teacher_model(tgt_model, msrc_model, keep_rate=0.99)
        msrc_model.load_state_dict(new_msrc_dict)

        if max_train_steps > 0 and step_idx >= max_train_steps:
            break
        
    pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(val_loader, model, criterion, save_dir=None, img_ids=None):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        # Thêm biến i để duyệt qua index
        for i, (input, target, meta) in enumerate(val_loader): 
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            loss = criterion(output, target)
            iou,dice = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            # ========================================================
            # ĐOẠN CODE LƯU ẢNH (CHỈ CHẠY NẾU CÓ TRUYỀN SAVE_DIR)
            # ========================================================
            if save_dir is not None:
                import cv2
                import numpy as np
                # Chuyển output thành xác suất và tạo mask nhị phân
                pred_prob = torch.sigmoid(output)
                pred_mask = (pred_prob > 0.5).float()
                
                # Ép kiểu về Numpy array [0, 255]
                mask_np = pred_mask[0].squeeze().cpu().numpy()
                final_mask = (mask_np * 255).astype(np.uint8)

                # Lấy tên file gốc hoặc dùng số thứ tự i
                img_name = img_ids[i] if img_ids is not None else f"pred_{i}"
                save_path = os.path.join(save_dir, f"{img_name}.png")
                cv2.imwrite(save_path, final_mask)
            # ========================================================

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])

def main():
    args = parse_args()

    config_file = "config_" + args.target
    config_path = 'models/%s/%s.yml' % (args.source, config_file)
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Expected a source config for target '{args.target}'."
        )
    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if isinstance(config, dict):
        config = {str(key).lstrip('\ufeff'): value for key, value in config.items()}

    # Federated-generated configs include architecture/data fields but can omit
    # optimization/runtime keys needed by tt_sfuda_2d. Fill safe defaults here.
    config.setdefault('name', args.source)
    config.setdefault('num_workers', 0)
    config.setdefault('lr', 1e-4)
    config.setdefault('weight_decay', 1e-4)
    config.setdefault('loss', 'BCEDiceLoss')
    config.setdefault('stage1', 1)
    config.setdefault('stage2', 1)
    if args.stage1 > 0:
        config['stage1'] = args.stage1
    if args.stage2 > 0:
        config['stage2'] = args.stage2
    adapt_lr = config['lr'] * args.adapt_lr_scale
    print(f"Adaptation config: stage1={config['stage1']}, stage2={config['stage2']}, "
          f"adapt_lr={adapt_lr:.2e} (base={config['lr']:.2e} × scale={args.adapt_lr_scale}), "
          f"pseudo_thresh={args.pseudo_thresh}")

    train_img_ids = glob(os.path.join('inputs', 'inputs', args.target, 'train','images', '*' + config['img_ext']))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]

    val_img_ids = glob(os.path.join('inputs', 'inputs', args.target, 'test','images', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]

    train_transform = A.Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', 'inputs', args.target, 'train','images'),
        mask_dir=os.path.join('inputs', 'inputs', args.target, 'train','masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform,
        missing_mask_strategy='zeros')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)

    val_transform = A.Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', 'inputs', args.target,'test', 'images'),
        mask_dir=os.path.join('inputs', 'inputs', args.target,'test', 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    print("Creating model %s...!!!" % config['arch'])
    print("Loading source trained model...!!!")
    msrc_model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    source_model_path = 'models/%s/model.pth' % config['name']
    if not os.path.exists(source_model_path):
        raise FileNotFoundError(
            f"Source checkpoint not found: {source_model_path}\n"
            "Run source training first, for example:\n"
            "  e:/Python/.venv/Scripts/python.exe train_source.py --dataset greenhouse_clean --epochs 50 --img-ext .jpg --missing-mask-strategy error"
        )
    msrc_model.load_state_dict(torch.load(source_model_path, map_location=device))
    msrc_model.to(device)
    msrc_model.train()
    print("Sucessfully loaded source trained model...!!!")

    tgt_model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    tgt_model.load_state_dict(torch.load(source_model_path, map_location=device))
    tgt_model.to(device)
    tgt_model.train()

    src_params = filter(lambda p: p.requires_grad, msrc_model.parameters())
    src_optimizer = optim.Adam(src_params, lr=adapt_lr, weight_decay=config['weight_decay'])

    tgt_params = filter(lambda p: p.requires_grad, tgt_model.parameters())
    tgt_optimizer = optim.Adam(tgt_params, lr=adapt_lr, weight_decay=config['weight_decay'])

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    
    pseudo_model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    pretrained_dict = msrc_model.state_dict()
    pseudo_model.load_state_dict(pretrained_dict)
    pseudo_model.to(device)
    pseudo_model.eval()

    criterion = losses.__dict__[config['loss']]().to(device)
    
    print("")
    print("Performing source only model evaluation...!!!")
    val_log = validate(val_loader, msrc_model, criterion)
    print('Source_only dice: %.4f' % (val_log['dice']))

    print("")
    print("Target specific adaptation...!!!")
    if args.max_train_steps > 0:
        print(f"Quick adaptation mode: max_train_steps={args.max_train_steps} per epoch")
    for epoch in range(config['stage1']):
        train_log = sfuda_target(
            config,
            train_loader,
            pseudo_model,
            msrc_model,
            criterion,
            src_optimizer,
            max_train_steps=args.max_train_steps,
            pseudo_thresh=args.pseudo_thresh,
        )
        print('train_loss %.4f - train_iou %.4f' % (train_log['loss'], train_log['iou']))

    msrc_model.eval()
    pretrained_dict = msrc_model.state_dict()
    tgt_model.load_state_dict(pretrained_dict)
    tgt_model.to(device)
    tgt_model.train()

    print("")
    print("Target model refinement...!!!")
    if args.max_train_steps > 0:
        print(f"Quick refinement mode: max_train_steps={args.max_train_steps} per epoch")
    for epoch in range(config['stage2']):
        train_log = sfuda_task(
            train_loader,
            msrc_model,
            tgt_model,
            criterion,
            tgt_optimizer,
            max_train_steps=args.max_train_steps,
            const_loss_weight=args.const_loss_weight,
        )
        print('refine_loss %.4f - refine_iou %.4f' % (train_log['loss'], train_log['iou']))

    print("")
    print("Performing adapted target model evaluation...!!!")
    
    # Save outputs with target-specific names to avoid dataset mixups.
    output_img_dir = f"results_{args.target}_masks"
    os.makedirs(output_img_dir, exist_ok=True)
    
    # 2. Gọi hàm validate mới (có truyền thêm thư mục và tên ảnh)
    val_log = validate(val_loader, tgt_model, criterion, save_dir=output_img_dir, img_ids=val_img_ids)
    
    adapted_model_path = f"adapted_target_model_{args.target}.pth"
    torch.save(tgt_model.state_dict(), adapted_model_path)
    
    print("\n-----------------------------------------------------------")
    print(f" Saved predicted masks to: {output_img_dir}")
    print(f" Saved adapted model to: {adapted_model_path}")
    print("-----------------------------------------------------------\n")
    # -----------------------------------

    print('Adapted target model dice: %.4f' % (val_log['dice']))


if __name__ == '__main__':
    main()
