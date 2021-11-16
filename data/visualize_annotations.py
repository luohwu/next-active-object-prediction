import cv2
import pandas as pd
from opt import *
import os
from ast import literal_eval


def visualize(par_video_id):
    par_id=par_video_id[0:3]
    video_id=par_video_id[3:]
    annos_file_path=os.path.join(args.data_path,annos_path,f'nao_{par_video_id}.csv')
    if os.path.exists(annos_file_path):
        print(annos_file_path)
        annos=pd.read_csv(annos_file_path)
        for i in range(0,annos.shape[0]):
            nao_bbox=literal_eval(annos.iloc[i]['nao_bbox_resized'])
            frame_id=annos.iloc[i]['frame']
            frame_name=f'frame_{str(frame_id).zfill(10)}.jpg'
            image_file_path=os.path.join(args.data_path,frames_path,par_id,video_id,frame_name)
            assert os.path.exists(image_file_path),f'file not exists: {image_file_path}'
            image=cv2.imread(image_file_path)
            cv2.rectangle(image,(nao_bbox[0],nao_bbox[1]),(nao_bbox[2],nao_bbox[3]),(0,255,0),3)
            cv2.imshow(f'{frame_name}',image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__=='__main__':
    par_video_id_list=['P12P12_01','P12P12_02', 'P12P12_04' ]
    for par_video_id in par_video_id_list:
        visualize(par_video_id)