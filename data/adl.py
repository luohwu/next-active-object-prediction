# @File : epic.py.py
# @Time : 2019/9/24
# @Email : jingjingjiang2017@gmail.com

import math
import os
import time
import pickle
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageOps
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from opt import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import  cv2
# train_video_id = list(train_video_id) + list(val_video_id)
# print(len(train_video_id))
# train_video_id = {'P01P01_01', 'P01P01_02', 'P01P01_03', 'P01P01_04',
#                   'P01P01_05', 'P01P01_06', 'P01P01_07', 'P01P01_08',
#                   'P01P01_09', 'P01P01_10', 'P01P01_16', 'P01P01_17',
#                   'P01P01_18', 'P01P01_19', 'P02P02_01', 'P02P02_02',
#                   'P02P02_03', 'P02P02_04', 'P02P02_05', 'P02P02_06',
#                   'P02P02_07', 'P02P02_08', 'P02P02_09', }
# train_video_id = {'P01P01_01', 'P01P01_02'}
# val_video_id = {'P01P01_01'}

def generate_pseudo_track_id(annos):
    video_id = annos.id[0]
    annos.loc[:, 'pseudo_track_id'] = -1
    track_id_ = 0
    for label in annos.label.unique():
        anno_ = annos[annos['label'] == label]
        if anno_.shape[0] <= 3:
            # print(label)
            annos.loc[annos['label'] == label, 'pseudo_track_id'] = \
                video_id + '_' + str(track_id_).zfill(3)
            track_id_ += 1
        else:
            # print(f'{label}: {anno_.shape[0]}')
            # frame_1 = anno_.iloc[0, 0]
            for j, frame in enumerate(anno_.frame):
                if j == 0:
                    annos.loc[anno_.index[0], 'pseudo_track_id'] = \
                        video_id + '_' + str(track_id_).zfill(3)
                else:
                    if (frame - anno_.iloc[j - 1, 0]) < 90:
                        annos.loc[anno_.index[j], 'pseudo_track_id'] = \
                            video_id + '_' + str(track_id_).zfill(3)
                    else:
                        track_id_ += 1
                        annos.loc[anno_.index[j], 'pseudo_track_id'] = \
                            video_id + '_' + str(track_id_).zfill(3)
            track_id_ += 1


def check_pseudo_track_id(annos):
    video_id = annos.id[0]
    annos.loc[:, 'pseudo_track_id'] = -1
    track_id_ = 0
    for label in annos.label.unique():
        anno_ = annos[annos['label'] == label]
        if anno_.shape[0] <= 3:
            # print(label)
            annos.loc[annos['label'] == label, 'pseudo_track_id'] = track_id_
            track_id_ += 1
        else:
            print(f'{label}: {anno_.shape[0]}')
            # frame_1 = anno_.iloc[0, 0]
            for j, frame in enumerate(sorted(anno_.frame)):
                if j == 0:
                    annos.loc[anno_.index[0], 'pseudo_track_id'] = track_id_
                else:
                    if (frame - anno_.iloc[j - 1, 0]) < 90:
                        annos.loc[anno_.index[j], 'pseudo_track_id'] = track_id_
                    else:
                        track_id_ += 1
                        annos.loc[anno_.index[j], 'pseudo_track_id'] = track_id_
            track_id_ += 1


def check_data_annos(args):
    df_items = pd.DataFrame(columns=['img_file', 'pseudo_track_id',
                                     'nao_bbox', 'label'])

    for video_id in sorted(train_video_id):
        start = time.process_time()
        img_path = os.path.join(args.data_path, frames_path,
                                str(video_id)[:3], str(video_id)[3:])

        anno_name = 'nao_' + video_id + '.csv'
        anno_path = os.path.join(args.data_path, annos_path, anno_name)
        annos = pd.read_csv(anno_path, converters={"nao_bbox": literal_eval})

        check_pseudo_track_id(annos)  # 生成track_id

        annos.insert(loc=5, column='img_file', value=0)
        for index in annos.index:
            img_file = img_path + '/' + str(annos.loc[index, 'frame']).zfill(
                10) + '.jpg'
            annos.loc[index, 'img_file'] = img_file

        annos_df = pd.DataFrame(annos, columns=['img_file', 'pseudo_track_id',
                                                'nao_bbox', 'label'])
        df_items = df_items.append(annos_df, ignore_index=False)

        end = time.process_time()
        print(f'finished video {video_id}, time is {end - start}')

    # 生成sequence data
    for idx, pt_id in enumerate(sorted(df_items.pseudo_track_id.unique())):
        df_items.loc[df_items.pseudo_track_id == pt_id, 'bs_idx'] = str(idx)

    print('================================================================')
    return df_items



# change dataframes [frame	id	label	nao_bbox] to ['img_file', 'pseudo_track_id',
#                                      'nao_bbox', 'label', 'bs_idx']
def make_sequence_dataset(mode='train'):

    #val is the same as test
    if mode=='all':
        par_video_id_list=ids_adl
    elif mode=='train':
        par_video_id_list = train_video_id
    else:
        par_video_id_list = test_video_id
    # par_video_id_list=train_video_id if mode=='train' else test_video_id

    print(f'start load {mode} data, size: {len(par_video_id_list)}')
    df_items = pd.DataFrame()
    for video_id in sorted(par_video_id_list):
        anno_name = 'nao_' + video_id + '.csv'
        anno_path = os.path.join(args.data_path, annos_path, anno_name)
        if os.path.exists(anno_path):
            # start = time.process_time()
            img_path = os.path.join(args.data_path, frames_path,video_id)

            annos = pd.read_csv(anno_path,
                                converters={"nao_bbox": literal_eval,
                                            "nao_bbox_resized": literal_eval})

            if not annos.empty:
                generate_pseudo_track_id(annos)  # 生成track_id

                # annos.insert(column='img_file', value=0)

                for index in annos.index:
                    frame_id=annos.loc[index,'frame']
                    img_file=os.path.join(img_path,f'frame_{str(frame_id).zfill(10)}.jpg')
                    annos.loc[index, 'img_file'] = img_file
                    # resized nao_bbox!
                annos_subset=annos[['img_file',  'pseudo_track_id',
                                                 'nao_bbox_resized', 'label','hand_bbox']]
                # annos_df = pd.DataFrame(annos,
                #                         columns=['img_file',  'pseudo_track_id',
                #                                  'nao_bbox_resized', 'label','hand_bbox'])
                df_items = df_items.append(annos_subset)

            # end = time.process_time()
            # df_items=df_items.rename(columns={'nao_bbox_resized':'nao_bbox'})
            # print(f'finished video {video_id}, time is {end - start}')

    # 生成sequence data
    for idx, pt_id in enumerate(sorted(df_items.pseudo_track_id.unique())):
        df_items.loc[df_items.pseudo_track_id == pt_id, 'bs_idx'] = str(idx)

    df_items = df_items.rename(columns={'nao_bbox_resized': 'nao_bbox'})
    print('finished')
    print('=============================================================')
    return df_items



class AdlDatasetV2(Dataset):
    def __init__(self, mode='train'):
        self.mode=mode
        self.args = args
        self.crop = transforms.RandomCrop((args.img_resize[0],
                                           args.img_resize[1]))
        self.transform_label = transforms.ToTensor()

        self.static_hm = self.gen_static_hand_dm()
        self.data = make_sequence_dataset(mode)


        # print(f'{mode} data: {self.data.shape[0]}')

        # # pandas的shuffle
        self.data = self.data.sample(frac=1).reset_index(drop=True)

        if args.normalize:
            self.transform = transforms.Compose([  # [h, w]
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # ImageNet,
                # ,AddGaussianNoise(0., 0.5 if self.mode=='train' else 0)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

    def __getitem__(self, item):
        df_item = self.data.iloc[item, :]

        img_file = df_item.img_file
        img = Image.open(img_file).convert('RGB')

        # nao_bbox = [x1, y1, x2, y2]  bbox = df_item.nao_bbox
        mask = self.generate_mask(img, df_item.nao_bbox)
        mask = Image.fromarray(mask)
        # hand_hm = self.hand_hms[img_file]
        # print(df_item.hand_bbox)
        if df_item.hand_bbox is np.nan:
            hand_hm = self.static_hm
        else:
            hand_bbox=literal_eval(df_item.hand_bbox)
            # print(hand_bbox)
            hand_hm = self.generate_hand_hm(img, hand_bbox)

        hand_hm = Image.fromarray(hand_hm)


        img = img.resize((self.args.img_resize[1],
                          self.args.img_resize[0]))

        mask = mask.resize((self.args.img_resize[1],
                            self.args.img_resize[0]))
        hand_hm = hand_hm.resize((self.args.img_resize[1],
                            self.args.img_resize[0]))

        img = self.transform(img)
        mask = self.transform_label(mask)[0, :, :]
        hand_hm = self.transform_label(hand_hm)

        return img, mask, hand_hm

    def __len__(self):  # batch迭代的次数与其有关
        return self.data.shape[0]

    @staticmethod
    def generate_mask(img, bbox):
        mask = np.zeros((img.size[1], img.size[0]), dtype=np.float32)
        mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1

        return mask

    @staticmethod
    def generate_hand_hm(img, hand_bbox):

        im = np.zeros((img.size[0], img.size[1]))

        if len(hand_bbox) > 0:
            points = []
            for box in hand_bbox:
                points.append((box[0], box[1]))
                # points.append((box[0], box[3]))
                points.append((box[1], box[1]))
                # points.append((box[1], box[3]))
            points = np.array(points).transpose()
            im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
            im = ndimage.gaussian_filter(im, sigma=img.size[0] / (
                    4. * points.shape[1]))
            im = (im - im.min()) / (im.max() - im.min())

        return im.transpose()


    @staticmethod
    def gen_static_hand_dm():  #
        im = np.zeros((1280, 960))
        points = [(459, 465), (896, 465)]

        points = np.array(points).transpose()
        im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
        im = ndimage.gaussian_filter(im, sigma=1280 / (4. * points.shape[1]))
        im = (im - im.min()) / (im.max() - im.min())

        return im.transpose()



    def generate_hms(self):
        hand_hms_dict = {}
        for index in self.data.index:
            if index % 100 == 0:
                print(f'index: {index}')
            # df_item = self.data.loc[index, :]
            img_file = self.data.loc[index, 'img_file']
            hand_bbox = self.data.loc[index, 'hand_bbox']
            # img = Image.open(img_file).convert('RGB')

            if hand_bbox is None:
                hand_hm = self.static_hm
            else:
                img = Image.open(img_file).convert('RGB')
                hand_hm = self.generate_hand_hm(img, hand_bbox)

            hand_hm = Image.fromarray(hand_hm)
            hand_hm = hand_hm.resize((self.args.img_resize[1],
                                      self.args.img_resize[0]))
            hand_hms_dict[img_file] = hand_hm

        # save_file = open(os.path.join(self.args.data_path,
        #                               f'{self.args.mode}_hand_hms.pkl'),
        #                  'wb')
        # pickle.dump(hand_hms_dict, save_file)
        # save_file.close()




def show(imgs, masks):

    for img in imgs:
        for idx in range(img.shape[0]):
            img_ = img[idx, :, :, :].cpu().numpy()
            mask_ = masks[0, idx, :, :].cpu().numpy()
            img_mask_ = (img_ * np.tile(mask_, (3, 1, 1))).transpose(1, 2, 0)

            plt.imshow(img_mask_)
            plt.show()


if __name__ == '__main__':
    # check_data_annos(args)
    # train_dataset = EpicDataset(args)

    train_dataset = AdlDatasetV2(mode='train')

    train_dataloader = DataLoader(train_dataset, batch_size=4,
                                  num_workers=3, shuffle=False)
    print(len(train_dataloader.dataset),train_dataset.__len__())
    # for data in train_dataloader:
    it=iter(train_dataloader)
    nao_bbox_list=[]
    img,mask,hand_hm=next(it)
    mask_s=(mask.data.cpu().squeeze(0))
    print(mask.shape)
    print(mask_s.shape)
    # train_dataset.data.to_csv('/media/luohwu/T7/dataset/ADL/test.csv',index=False)
    # for i in range(100):
    #     img, mask, hand_hm = train_dataset.__getitem__(i)
    #     hand_hm=hand_hm.squeeze(0)
    #     img_numpy=img.numpy().transpose(1,2,0)
    #     cv2.imshow('image',img_numpy)
    #     cv2.imshow('image_mask', mask.numpy())
    #     cv2.imshow('hand_mask', hand_hm.numpy())
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     print(img.shape)
    #     print(mask.shape)
    #     print(hand_hm.shape)
    # train_dataset.generate_img_mask_pair()
    # train_dataset.generate_hm()
    # train_dataset = EpicSequenceDataset(args)
    # train_dataloader = DataLoader(train_dataset)
    # train_dataloader = DataLoader(train_dataset, batch_size=4,
    #                               num_workers=3, shuffle=False)
    # # sequence_lens = []
    # for i, data in enumerate(train_dataloader):
    #     img, mask, hand_hm = data
    #     # sequence_lens.append(img.shape[0])
    #     # show(img, mask)
    #     # print(img.shape)
