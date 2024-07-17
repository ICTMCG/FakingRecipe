import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import h5py
import json



class FakingRecipe_Dataset(Dataset):
    def __init__(self, vid_path,dataset):
        self.dataset=dataset
        if dataset=='fakesv': 
            self.data_all = pd.read_json('./data/FakeSV/fakesv_data.json',orient='records',dtype=False,lines=True)
            self.vid=[]
            with open(vid_path, "r") as fr:
                for line in fr.readlines():
                    self.vid.append(line.strip())
            self.data = self.data_all[self.data_all.video_id.isin(self.vid)]
            self.data.reset_index(inplace=True)

            self.ocr_pattern_fea_path='./data/FakeSV/preprocess_ocr/sam'
            self.ocr_phrase_fea_path='./data/FakeSV/preprocess_ocr/ocr_phrase_fea.pkl'
            with open(self.ocr_phrase_fea_path, 'rb') as f:
                self.ocr_phrase = torch.load(f)

            self.text_semantic_fea_path='./data/FakeSV/preprocess_text/sem_text_fea.pkl'
            with open(self.text_semantic_fea_path, 'rb') as f:
                self.text_semantic_fea = torch.load(f)

            self.text_emo_fea_path='./data/FakeSV/preprocess_text/emo_text_fea.pkl'
            with open(self.text_emo_fea_path, 'rb') as f:
                self.text_emo_fea = torch.load(f)

            self.audio_fea_path='./data/FakeSV/preprocess_audio'
            self.visual_fea_path='./data/FakeSV/preprocess_visual'
        elif dataset=='fakett':
            self.data_all=pd.read_json('./data/FakeTT/fakett_data.json',orient='records',lines=True,dtype={'video_id': str})
            self.vid=[]
            with open(vid_path, "r") as fr:
                for line in fr.readlines():
                    self.vid.append(line.strip())
            self.data = self.data_all[self.data_all.video_id.isin(self.vid)]
            self.data.reset_index(inplace=True)
            self.ocr_pattern_fea_path='./data/FakeTT/preprocess_ocr/sam'

            self.ocr_phrase_fea_path='./data/FakeTT/preprocess_ocr/ocr_phrase_fea.pkl'
            with open(self.ocr_phrase_fea_path, 'rb') as f:
                self.ocr_phrase = torch.load(f)

            self.text_semantic_fea_path='./data/FakeTT/preprocess_text/sem_text_fea.pkl'
            with open(self.text_semantic_fea_path, 'rb') as f:
                self.text_semantic_fea = torch.load(f)

            self.text_emo_fea_path='./data/FakeTT/preprocess_text/emo_text_fea.pkl'
            with open(self.text_emo_fea_path, 'rb') as f:
                self.text_emo_fea = torch.load(f)

            self.audio_fea_path='./data/FakeTT/preprocess_audio'
            self.visual_fea_path='./data/FakeTT/preprocess_visual'

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']
        label = 1 if item['annotation']=='fake' else 0
        fps=torch.tensor(item['fps'])
        total_frame=torch.tensor(item['frame_count'])
        visual_time_region=torch.tensor(item['transnetv2_segs'])
        label = torch.tensor(label)

        all_phrase_semantic_fea=self.text_semantic_fea['last_hidden_state'][vid]
        all_phrase_emo_fea=self.text_emo_fea['pooler_output'][vid]

        v_fea_path=os.path.join(self.visual_fea_path,vid+'.pkl')
        raw_visual_frames=torch.tensor(torch.load(open(v_fea_path,'rb'))) 

        a_fea_path=os.path.join(self.audio_fea_path,vid+'.pkl')
        raw_audio_emo=torch.load(open(a_fea_path,'rb')) #1*768

        ocr_pattern_fea_file_path=os.path.join(self.ocr_pattern_fea_path,vid,'r0.pkl') 
        ocr_pattern_fea=torch.tensor(torch.load(open(ocr_pattern_fea_file_path,'rb'))) 

        ocr_phrase_fea=self.ocr_phrase['ocr_phrase_fea'][vid] 
        ocr_time_region=self.ocr_phrase['ocr_time_region'][vid] 

        v_fea_path=os.path.join(self.visual_fea_path,vid+'.pkl')
        raw_visual_frames=torch.tensor(torch.load(open(v_fea_path,'rb')))

        return {
            'vid': vid,
            'label': label,
            'fps': fps,
            'total_frame': total_frame,
            'all_phrase_semantic_fea': all_phrase_semantic_fea,
            'all_phrase_emo_fea': all_phrase_emo_fea,
            'raw_visual_frames': raw_visual_frames,
            'raw_audio_emo': raw_audio_emo,
            'ocr_pattern_fea': ocr_pattern_fea,
            'ocr_phrase_fea': ocr_phrase_fea,
            'ocr_time_region': ocr_time_region,
            'visual_time_region': visual_time_region
        }

def pad_frame_sequence(seq_len,lst):
    attention_masks = []
    result=[]
    for video in lst:
        video=torch.FloatTensor(video)
        ori_len=video.shape[0]
        if ori_len>=seq_len:
            gap=ori_len//seq_len
            video=video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video=torch.cat((video,torch.zeros([seq_len-ori_len,video.shape[1]],dtype=torch.float)),dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len-ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)

def pad_frame_by_seg(seq_len,lst,seg):
    result=[]
    seg_indicators = []
    sampled_seg=[]
    for i in range(len(lst)):
        video=lst[i]
        v_sampled_seg=[]
        video=torch.FloatTensor(video)
        ori_len=video.shape[0]
        seg_video=seg[i]
        seg_len=len(seg_video)
        if seg_len>=seq_len:
            gap=seg_len//seq_len
            seg_video=seg_video[::gap][:seq_len]
            sample_index=[]
            sample_seg_indicator=[]
            for j in range(len(seg_video)):
                v_sampled_seg.append(seg_video[j])
                if seg_video[j][0]==seg_video[j][1]:
                    sample_index.append(seg_video[j][0])
                else:
                    sample_index.append(np.random.randint(seg_video[j][0],seg_video[j][1]))
                sample_seg_indicator.append(j)
            video=video[sample_index]
            mask=sample_seg_indicator
        else:
            if ori_len<seq_len:
                video=torch.cat((video,torch.zeros([seq_len-ori_len,video.shape[1]],dtype=torch.float)),dim=0)

                mask=[]
                for j in range(len(seg_video)):
                    v_sampled_seg.append(seg_video[j])
                    mask.extend([j]*(seg_video[j][1]-seg_video[j][0]+1))
                mask.extend([-1]*(seq_len-len(mask)))
                
            else:

                sample_index=[]
                sample_seg_indicator=[]
                seg_len=[(x[1]-x[0])+1 for x in seg_video]
                sample_ratio=[seg_len[i]/sum(seg_len) for i in range(len(seg_len))]
                sample_len=[seq_len*sample_ratio[i] for i in range(len(seg_len))]
                sample_per_seg=[int(x)+1 if x<1 else int(x) for x in sample_len]

                sample_per_seg=[x if x<=seg_len[i] else seg_len[i] for i,x in enumerate(sample_per_seg)]
                additional_sample=sum(sample_per_seg)-seq_len
                if additional_sample>0:
                    idx=0
                    while additional_sample>0:
                        if idx==len(sample_per_seg):
                            idx=0
                        if sample_per_seg[idx]>1:
                            sample_per_seg[idx]=sample_per_seg[idx]-1
                            additional_sample=additional_sample-1
                        idx+=1
                        
                elif additional_sample<0:
                    idx=0
                    while additional_sample<0:
                        if idx==len(sample_per_seg):
                            idx=0
                        if seg_len[idx]-sample_per_seg[idx]>=1:
                            sample_per_seg[idx]=sample_per_seg[idx]+1
                            additional_sample=additional_sample+1
                        idx+=1
                        
                
                for seg_idx in range(len(sample_per_seg)):
                    sample_seg_indicator.extend([seg_idx]*sample_per_seg[seg_idx])
                
                
                for j in range(len(seg_video)):
                    v_sampled_seg.append(seg_video[j])
                    if sample_per_seg[j]==seg_len[j]:
                        sample_index.extend(np.arange(seg_video[j][0],seg_video[j][1]+1))
                        
                    else:
                        sample_index.extend(np.sort(np.random.randint(seg_video[j][0],seg_video[j][1]+1,sample_per_seg[j])))
                        
                sample_index=np.array(sample_index)
                sample_index=np.sort(sample_index)
                video=video[sample_index]
                batch_sample_seg_indicator=np.array(sample_seg_indicator)
                mask=batch_sample_seg_indicator
                v_sampled_seg.sort(key=lambda x:x[0])

        result.append(video)
        mask = torch.IntTensor(mask)
        sampled_seg.append(v_sampled_seg) 
        seg_indicators.append(mask)
    return torch.stack(result), torch.stack(seg_indicators),sampled_seg

def pad_segment(seg_lst,target_len):
    for sl_idx in range(len(seg_lst)):
        for s_idx in range(len(seg_lst[sl_idx])):
            seg_lst[sl_idx][s_idx]=torch.tensor(seg_lst[sl_idx][s_idx])
        if len(seg_lst[sl_idx])<target_len:
            seg_lst[sl_idx].extend([torch.tensor([-1,-1])]*(target_len-len(seg_lst[sl_idx])))
        else:
            seg_lst[sl_idx]=seg_lst[sl_idx][:target_len] 
        seg_lst[sl_idx]=torch.stack(seg_lst[sl_idx])

    return torch.stack(seg_lst)

def pad_unnatural_phrase(phrase_lst,target_len): 
    for pl_idx in range(len(phrase_lst)):
        if len(phrase_lst[pl_idx])<target_len:
            phrase_lst[pl_idx]=torch.cat((phrase_lst[pl_idx],torch.zeros([target_len-len(phrase_lst[pl_idx]),phrase_lst[pl_idx].shape[1]],dtype=torch.long)),dim=0)           
        else:
            phrase_lst[pl_idx]=phrase_lst[pl_idx][:target_len]
    return torch.stack(phrase_lst)



def collate_fn_FakeingRecipe(batch):
    num_visual_frames=83
    num_segs=83
    num_phrase=80

    vid = [item['vid'] for item in batch]
    label = torch.stack([item['label'] for item in batch])
    all_phrase_semantic_fea = [item['all_phrase_semantic_fea'] for item in batch] #batch*512*768
    all_phrase_emo_fea = torch.stack([item['all_phrase_emo_fea'] for item in batch]) #batch*768

    raw_visual_frames = [item['raw_visual_frames'] for item in batch]
    raw_audio_emo = [item['raw_audio_emo'] for item in batch] 
    fps = torch.stack([item['fps'] for item in batch])
    total_frame = torch.stack([item['total_frame'] for item in batch])

    content_visual_frames, _ = pad_frame_sequence(num_visual_frames,raw_visual_frames)
    raw_audio_emo = torch.cat(raw_audio_emo,dim=0) #batch*768

    all_phrase_semantic_fea=[x if x.shape[0]==512 else torch.cat((x,torch.zeros([512-x.shape[0],x.shape[1]],dtype=torch.float)),dim=0) for x in all_phrase_semantic_fea] #batch*512*768
    all_phrase_semantic_fea=torch.stack(all_phrase_semantic_fea)

    ocr_pattern_fea = torch.stack([item['ocr_pattern_fea'] for item in batch]) #batch*4096
    ocr_phrase_fea = [item['ocr_phrase_fea'] for item in batch] #batch*seq_len*512
    ocr_time_region = [item['ocr_time_region'] for item in batch] #batch*seq_len*2

    visual_time_region = [item['visual_time_region'] for item in batch] #batch*seg_len*2

    visual_frames_fea,visual_frames_seg_indicator,sampled_seg=pad_frame_by_seg(num_visual_frames,raw_visual_frames,visual_time_region)
    visual_seg_paded=pad_segment(sampled_seg,num_segs)

    ocr_phrase_fea=pad_unnatural_phrase(ocr_phrase_fea,num_phrase)
    ocr_time_region=pad_unnatural_phrase(ocr_time_region,num_phrase)

    return {
        'vid': vid,
        'label': label,
        'fps': fps,
        'total_frame': total_frame,
        'all_phrase_semantic_fea': all_phrase_semantic_fea,
        'all_phrase_emo_fea': all_phrase_emo_fea,
        'raw_visual_frames': content_visual_frames,
        'raw_audio_emo': raw_audio_emo,
        'ocr_pattern_fea': ocr_pattern_fea,
        'ocr_phrase_fea': ocr_phrase_fea,
        'ocr_time_region': ocr_time_region,
        'visual_frames_fea': visual_frames_fea,
        'visual_frames_seg_indicator': visual_frames_seg_indicator,
        'visual_seg_paded': visual_seg_paded
    }
