# @File : train.py
# @Time : 2019/10/18 
# @Email : jingjingjiang2017@gmail.com 

import os
from comet_ml import Experiment
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

experiment = Experiment(
    api_key="wU5pp8GwSDAcedNSr68JtvCpk",
    project_name="baseline",
    workspace="thesisproject",
    auto_metric_logging=False
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

experiment.log_parameters(args.__dict__)
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
print('current graphics card is:')
os.system('lspci | grep VGA')

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
        if args.original_split:
            train_data = EpicDatasetV2('train')
            val_data = EpicDatasetV2('val')
        else:
            all_data=EpicDatasetV2('all')
            train_data, val_data=torch.utils.data.random_split(all_data,[8589,3000])
        train_dataloader = DataLoader(train_data, batch_size=args.bs,
                                      shuffle=True, num_workers=4,
                                      pin_memory=True)
        

        val_dataloader = DataLoader(val_data,
                                    batch_size=args.bs,
                                    shuffle=True, num_workers=4,
                                    pin_memory=True)
    else:
        if args.original_split:
            train_data = AdlDatasetV2('train')
            val_data = AdlDatasetV2('val')
        else:
            all_data=AdlDatasetV2('all')
            train_data,val_data=torch.utils.data.random_split(all_data,[1767,450])
        train_dataloader = DataLoader(train_data, batch_size=args.bs,
                                          shuffle=True, num_workers=4,
                                          pin_memory=True)
        

        val_dataloader = DataLoader(val_data,
                                    batch_size=args.bs,
                                    shuffle=True, num_workers=4,
                                    pin_memory=True)

    print(f'train data size: {len(train_data)}, val data size: {len(val_data)}')
    
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           betas=(0.9, 0.99),
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.8,
                                                     patience=5,
                                                     verbose=True,
                                                     min_lr=0.000001)
    
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
    criterion = nn.CrossEntropyLoss(class_weights)
    # criterion=CIOU_LOSS()
    # criterion = FocalLoss()
    
    train_args['ckpt_path'] = os.path.join(train_args['exp_path'],args.dataset,
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
        if epoch % epoch_save==0:
            checkpoint_path=os.path.join(train_args['ckpt_path'],f'model_epoch_{epoch}.pth')
            torch.save({'epoch':epoch,
                        'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict()},
                       checkpoint_path)
        print(f'train loss: {train_loss:.8f} | val loss:{val_loss:.8f}')
        experiment.log_metric("train_loss", train_loss, step=epoch)
        experiment.log_metric("val_loss", val_loss, step=epoch)

        # print(f"==================epoch :{epoch}/{train_args['epochs']+1}===============================================")


    # writer_train.close()
    # write_val.close()


def train(train_dataloader, model, criterion, optimizer, epoch, train_args):
    train_losses = 0.
    targets_all, predictions_all = [], []
    # curr_iter = (epoch - 1) * len(train_dataloader)

    for i, data in enumerate(train_dataloader, start=1):
        img, mask, hand_hm = data
        # img = Variable(img.float().cuda(args.device_ids[0]))
        img=img.float().to(device)
        hand_hm=hand_hm.float().to(device)
        # forward
        outputs = model(img, hand_hm)
        # print(f'output size:{outputs.shape}')
        # outputs = model(hand_hm)
        del img, hand_hm
        predictions_all.append(outputs.data.max(1)[1].cpu().numpy())  # outputs.data.max(1)[1] of shape [Batch_size, height, width]

        # works when batch_size=1
        targets_all.append(mask.data.cpu().squeeze(0))

        # loss = criterion(outputs, mask.long().cuda(args.device_ids[0]))
        loss = criterion(outputs, mask.long().to(device))
        # print(f'mask shape: {mask.shape}')
        # print(f'output shape: {outputs.shape}')

        del outputs, mask

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
    acc_, precision, recall, f1_score_ = compute_metrics(predictions_all,
                                                         targets_all)
    experiment.log_metric("train_acc_avg", acc_, step=epoch)
    experiment.log_metric("train_f1_avg", f1_score_, step=epoch)

    del targets_all, predictions_all
    return train_losses / len(train_dataloader)
    # print(f"[epoch {epoch}], avg train loss: {train_losses/len(train_dataloader):5f} ")
    # if i % train_args['print_every'] == 0:
    #     print(f"[epoch {epoch}], [iter {i} / {len(train_dataloader)}], "
    #           f"[train loss {train_losses / i:5f}]")



def val(val_dataloader, model, criterion, epoch, write_val):
    model.eval()
    val_loss = AverageMeter()
    targets_all, predictions_all = [], []
    loader_size = len(val_dataloader)
    for i, data in enumerate(val_dataloader):
        # print(f'{i}/{loader_size}')
        img, mask, hand_hm = data
        n = img.size(0)
        # img = Variable(img.float().cuda(args.device_ids[0]))
        # img = Variable(img.float().to(device))
        img=img.float().to(device)
        # hand_hm = Variable(hand_hm.float().cuda(args.device_ids[0]))
        # hand_hm = Variable(hand_hm.float().to(device))
        hand_hm=hand_hm.float().to(device)
        # mask = mask.long().cuda(args.device_ids[0])
        mask = mask.long().to(device)
        # forward
        outputs = model(img, hand_hm)
        # outputs = model(hand_hm)
        del img, hand_hm

        predictions_all.append(outputs.data.max(1)[1].cpu().numpy()) #outputs.data.max(1)[1] of shape [Batch_size, height, width]

        #works when batch_size=1
        targets_all.append(mask.data.cpu().squeeze(0))

        # loss = criterion(outputs.permute(0, 2, 3, 1).reshape([-1, 2]),
        #                  mask.flatten())
        # print(f'outputs size {outputs.shape},  mask size: {mask.shape}')
        loss = criterion(outputs, mask)
        val_loss.update(loss.item(), n)

        del outputs, mask

    acc_, precision, recall, f1_score_ = compute_metrics(predictions_all,
                                                         targets_all)
    experiment.log_metric("val_acc_avg", acc_, step=epoch)
    experiment.log_metric("val_f1_avg", f1_score_, step=epoch)

    del targets_all, predictions_all
    print(f'[epoch {epoch}], [val loss {val_loss.avg:5f}], [acc {acc_:5f}], '
          f'[precision {precision:5f}], [recall {recall:5f}], '
          f'[f1_score {f1_score_:5f}]')

    write_val.writelines(f"[epoch {epoch}], "
                         f"[acc {acc_:5f}], [precision {precision:5f}], "
                         f"[recall {recall:5f}], [f1_score {f1_score_:5f}]]\n")



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