import torch.utils.data as data 
import os 
import numpy as np 

# this function can be optimized 
def default_flist_reader(ply_data_dir, flist): 
    """
    flist format: pts_file seg_file label for each line 
    """
    ffiles = open(flist, 'r')
    lines = [line.rstrip() for line in ffiles.readlines()]

    pts_files = [os.path.join(ply_data_dir, line.split()[0]) for line in lines]
    seg_files = [os.path.join(ply_data_dir, line.split()[1]) for line in lines]
    labels = [line.split()[2] for line in lines]
    ffiles.close()

    all_data = [] 
    for pts_file_path, seg_file_path, label_id in zip(pts_files, seg_files, labels):
        all_data.append((pts_file_path, seg_file_path, label_id))

    return all_data # (pts_file_path, seg_file_path, label_id)

def default_loader(pts_file_path, seg_file_path):
    with open(pts_file_path, 'r') as f: 
        pts_str = [item.rstrip() for item in f.readlines()]
        pts = np.array([np.float32(s.split()) for s in pts_str], dtype=np.float32)

    with open(seg_file_path, 'r') as f:
        part_ids = np.array([int(item.rstrip()) for item in f.readlines()], dtype=np.uint8)

    return pts, part_ids

class PlyFileList(data.Dataset):
    def __init__(self, ply_data_dir, 
        test_ply_file_list_path, 
        label_id_pid2pid_in_set, 
        label_ids2ids, label_ids, 
        flist_reader=default_flist_reader, 
        transform=None, target_transform=None,
        loader=default_loader):

        self.ply_data_full_paths = flist_reader(ply_data_dir, test_ply_file_list_path)
        self.label_id_pid2pid_in_set = label_id_pid2pid_in_set
        self.label_ids2ids = label_ids2ids
        self.label_ids = label_ids
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader 

    def __getitem__(self, index):
        pts_file_path, seg_file_path, label_id = self.ply_data_full_paths[index]

        cur_gt_label = self.label_ids2ids[label_id]

        pts_data, part_ids= self.loader(pts_file_path, seg_file_path)
        # convert to seg_data
        seg_data = np.array([self.label_id_pid2pid_in_set[self.label_ids[cur_gt_label]+'_'+str(x)] for x in part_ids])

        if self.transform is not None:
            pts_data = self.transform(pts_data)
        if self.target_transform is not None:
            seg_data = self.target_transform(seg_data)

        return pts_data, seg_data, label_id

    def __len__(self):
        return len(self.ply_data_full_paths)