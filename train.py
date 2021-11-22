# @File : train.py
# @Time : 2019/10/18 
# @Email : jingjingjiang2017@gmail.com 

import os
from datetime import datetime

import torch.cuda
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader

from data.adl import AdlDatasetV2
from data.epic import EpicDatasetV2
from metrics.losses import *
from metrics.metric import *
from model.unet_resnet_hand_att import UNetResnetHandAtt
from opt import *
import tarfile
from metrics.CIOU import CIOU_LOSS

###########################################################################
# os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5'
# str_ids = args.gpu_ids.split(',')
# gpu_ids = []
# for str_id in str_ids:
#     id = int(str_id)
#     if id >= 0:
#         gpu_ids.append(id)
# # set gpu ids
# if len(gpu_ids) > 0:
#     torch.cuda.set_device(gpu_ids[0])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
###########################################################################


SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


exp_name = args.exp_name

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
# writer_train = SummaryWriter(os.path.join(args.exp_path, exp_name, 'logs',
#                                           f'train/{TIMESTAMP}'))
# writer_val = SummaryWriter(os.path.join(args.exp_path, exp_name, 'logs',
#                                         f'val/{TIMESTAMP}'))

multi_gpu = True if torch.cuda.device_count()>1 else False
print(f'using {torch.cuda.device_count()} GPUs')
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def main():
    model = UNetResnetHandAtt()
    for p in model.base_model.parameters():
        p.requires_grad = False
    # load parameters                      
    # model.load_state_dict(torch.load(
    #     os.path.join(args.exp_path, exp_name,
    #                  f'ckpts/model_epoch_{current_epoch}.pth')))

    
    # model.cuda(device=args.device_ids[0])
    if multi_gpu==True:
        model=nn.DataParallel(model)
    model=model.to(device)
    
    if args.dataset == 'EPIC':
        train_data = EpicDatasetV2('train')
        train_dataloader = DataLoader(train_data, batch_size=args.bs,
                                      shuffle=True, num_workers=2,
                                      pin_memory=True)
        

        val_data = EpicDatasetV2('val')
        val_dataloader = DataLoader(val_data,
                                    batch_size=args.bs,
                                    shuffle=True, num_workers=2,
                                    pin_memory=True)
    else:
        train_data = AdlDatasetV2('train')
        train_dataloader = DataLoader(train_data, batch_size=args.bs,
                                      shuffle=True, num_workers=2,
                                      pin_memory=True)
        

        val_data = AdlDatasetV2('val')
        val_dataloader = DataLoader(val_data,
                                    batch_size=args.bs,
                                    shuffle=True, num_workers=2,
                                    pin_memory=True)
    
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           betas=(0.9, 0.99),
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.8,
                                                     patience=5,
                                                     verbose=True,
                                                     min_lr=0.0000001)
    
    # if multi_gpu:
    #     optimizer = nn.DataParallel(optimizer)
    # else:
    # UNet的criterion
    if args.dataset == 'EPIC':
        # class_weights = torch.FloatTensor([1, 11.2]).cuda(args.device_ids[0])
        class_weights = torch.FloatTensor([1, 11.2]).to(device)
    else:
        # class_weights = torch.FloatTensor([1, 9.35]).cuda(args.device_ids[0])
        class_weights = torch.FloatTensor([1, 9.35]).to(device)
    # criterion = nn.CrossEntropyLoss(class_weights)
    criterion=CIOU_LOSS()
    # criterion = FocalLoss()
    
    train_args['ckpt_path'] = os.path.join(train_args['exp_path'],
                                           exp_name, 'ckpts/')
    if not os.path.exists(train_args['ckpt_path']):
        os.mkdir(train_args['ckpt_path'])
    
    write_val = open(os.path.join(train_args['ckpt_path'], 'val.txt'), 'w')

    train_loss_list=[]
    val_loss_list=[]
    current_epoch = 0
    epoch_save=50 if args.dataset=='EPIC' else 200
    for epoch in range(current_epoch + 1, train_args['epochs'] + 1):
        print(f"==================epoch :{epoch}/{train_args['epochs']}===============================================")
        train_loss = train(train_dataloader, model, criterion, optimizer, epoch, train_args)
        val_loss = val(val_dataloader, model, criterion, epoch, write_val)
        scheduler.step(val_loss)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        if epoch % epoch_save==0:
            checkpoint_path=os.path.join(train_args['ckpt_path'],f'model_epoch_{epoch}.pth')
            torch.save({'epoch':epoch,
                        'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'val_loss_list':val_loss_list,
                        'train_loss_list':train_loss_list},
                       checkpoint_path)
        print(f'train loss: {train_loss:.8f} | val loss:{val_loss:.8f}')
        # print(f"==================epoch :{epoch}/{train_args['epochs']+1}===============================================")


    # writer_train.close()
    # write_val.close()


def train(train_dataloader, model, criterion, optimizer, epoch, train_args):
    train_losses = 0.
    # curr_iter = (epoch - 1) * len(train_dataloader)
    
    for i, data in enumerate(train_dataloader, start=1):
        img, nao_bbox, hand_hm = data
        # img = Variable(img.float().cuda(args.device_ids[0]))
        img = Variable(img.float().to(device))
        # hand_hm = Variable(hand_hm.float().cuda(args.device_ids[0]))
        hand_hm = Variable(hand_hm.float().to(device))
        # forward
        outputs = model(img, hand_hm)
        # print(f'output size:{outputs.shape}')
        # outputs = model(hand_hm)
        del img, hand_hm
        
        # loss = criterion(outputs, mask.long().cuda(args.device_ids[0]))
        loss,_ = criterion(outputs, nao_bbox.to(device))
        # print(f'mask shape: {mask.shape}')
        # print(f'output shape: {outputs.shape}')

        # del outputs, mask
        del outputs, nao_bbox

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if multi_gpu:
        #     optimizer.module.step()
        # else:
        #     optimizer.step()
        train_losses += loss.item()

        # curr_iter += 1
        # writer_train.add_scalar("train_loss", train_losses / i, curr_iter)
    return train_losses/len(train_dataloader.dataset)
    # print(f"[epoch {epoch}], avg train loss: {train_losses/len(train_dataloader):5f} ")
        # if i % train_args['print_every'] == 0:
        #     print(f"[epoch {epoch}], [iter {i} / {len(train_dataloader)}], "
        #           f"[train loss {train_losses / i:5f}]")


def val(val_dataloader, model, criterion, epoch, write_val):
    model.eval()
    val_loss = AverageMeter()
    iou_recorder=AverageMeter()
    num_correct=0
    iou_threshold=0.5
    len_dataset=len(val_dataloader.dataset)
    # targets_all, predictions_all = [], []
    # loader_size=len(val_dataloader)
    for i,data in enumerate(val_dataloader):
        # print(f'{i}/{loader_size}')
        img, nao_bbox_gt, hand_hm = data
        n = img.size(0)
        # img = Variable(img.float().cuda(args.device_ids[0]))
        img = Variable(img.float().to(device))
        # hand_hm = Variable(hand_hm.float().cuda(args.device_ids[0]))
        hand_hm = Variable(hand_hm.float().to(device))
        # mask = mask.long().cuda(args.device_ids[0])
        # mask = mask.long().to(device)
        nao_bbox_gt=nao_bbox_gt.to(device)
        # forward
        outputs = model(img, hand_hm)
        # outputs = model(hand_hm)
        del img, hand_hm
        # print(f'output.data size : {outputs.data.shape}')
        # predictions_all.append(outputs.data.max(1)[1].cpu().numpy())
        # print(f'output.data selected size : {outputs.data.max(1)[1].shape}')
        # print(f'{type(mask)}')
        # targets_all.append(nao_bbox.data.cpu().squeeze(0))
        
        # loss = criterion(outputs.permute(0, 2, 3, 1).reshape([-1, 2]),
        #                  mask.flatten())
        # print(f'outputs size {outputs.shape},  mask size: {mask.shape}')
        loss,iou = criterion(outputs, nao_bbox_gt)
        num_correct+=torch.sum(iou>iou_threshold)

        val_loss.update(loss.item(), n)
        iou_recorder.update(torch.sum(iou).item(),n)
        del outputs, nao_bbox_gt
    
    # acc_, precision, recall, f1_score_ = compute_metrics(predictions_all,
    #                                                      targets_all)
    # del targets_all, predictions_all
    # print(f'[epoch {epoch}], [val loss {val_loss.avg:5f}], [acc {acc_:5f}], '
    #       f'[precision {precision:5f}], [recall {recall:5f}], '
    #       f'[f1_score {f1_score_:5f}]')
    print(f'[epoch {epoch}], [val loss {val_loss.avg:5f}], [IOU avg {iou_recorder.avg:5f}], [acc avg {num_correct/len_dataset}]')

    # write_val.writelines(f"[epoch {epoch}], "
    #                      f"[acc {acc_:5f}], [precision {precision:5f}], "
    #                      f"[recall {recall:5f}], [f1_score {f1_score_:5f}]]\n")
    write_val.writelines(f"[epoch {epoch}], [IOU avg {iou_recorder.avg:5f}],[acc avg {num_correct/len_dataset}] \n")
    
    # writer_val.add_scalar('val_loss', val_loss.avg, epoch)
    # writer_val.add_scalar('acc', acc_, epoch)
    # writer_val.add_scalar('precision', precision, epoch)
    # writer_val.add_scalar('recall', recall, epoch)
    # writer_val.add_scalar('f1_score', f1_score_, epoch)

    model.train()
    
    return val_loss.avg


if __name__ == '__main__':

    if args.euler:
        scratch_path=os.environ['TMPDIR']
        tar_path='/cluster/home/luohwu/dataset.tar.gz'
        assert os.path.exists(tar_path), f'file not exist: {tar_path}'
        print('extracting dataset from tar file')
        tar=tarfile.open(tar_path)
        tar.extractall(os.environ['TMPDIR'])
        tar.close()
        print('finished')
    # train_data = EpicDatasetV2('train')

    main()