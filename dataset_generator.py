from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np

class DS_generator(Sequence):

    def __init__(self,
                 x_train,
                 live_train,
                 batch_size=256,
                 mask_prob=0.0,
                 flip_prob=0.5,
                 increase_mask_prob=True, 
                 mask_prob_limit = 0.65):
        
        self.x_train = x_train
        self.live_train = live_train
        self.batch_size = batch_size
        self.mask_prob=mask_prob
        self.flip_prob=flip_prob
        self.delta_prob = (0.5 - mask_prob)/150
        self.increase_mask_prob = increase_mask_prob
        self.mask_prob_limit = mask_prob_limit


    def grid_mask(self, side, n):
        sq_0 = np.zeros((side, side, 3))
        sq_1 = np.full((side, side, 3), fill_value=1)
        row_0 = np.hstack([sq_0, sq_1] * n)
        row_1 = np.hstack([sq_1, sq_0] * n)
        if np.random.uniform(0,1) > 0.5:
            mask = np.vstack([row_0, row_1] * n)
        else:
            mask = np.vstack([row_1, row_0] * n)
        mask = np.array([mask for _ in range(self.batch_size)])
        return mask


    def apply_grid_mask(self, batch, offset_ratio=-0.05, side_ratio=0.225):
        img_h = 224
        img_w = 224
        side = round(img_h * side_ratio)

        if offset_ratio < 0:
            offset = 0
            mask_offset = - round(img_h * offset_ratio)
        else:
            offset = round(img_h * offset_ratio)
            mask_offset = 0
        n = (max(img_h, img_w) - offset) // side

        mask = self.grid_mask(side, n)
        masked_batch = batch.copy()
        masked_batch[:, offset:, offset:, :] = masked_batch[:, offset:, offset:, :] * mask[:, mask_offset:img_h-offset+mask_offset, mask_offset:img_w-offset+mask_offset, :]
        return masked_batch

		
    def on_epoch_end(self):
        if self.increase_mask_prob:
            self.mask_prob += self.delta_prob
            self.mask_prob = min(self.mask_prob, self.mask_prob_limit)


    def flip(self, batch):
        return batch[:,:,::-1,:]


    def __len__(self):                                      
        return len(self.x_train) // self.batch_size


    def __getitem__(self, idx):                                         

        indices_spoof = np.random.choice(len(self.x_train), size=int(self.batch_size // 4), replace=False)
        indices_live = np.random.choice(len(self.live_train), size=int(self.batch_size // (4/3)), replace=False)
        
        x_batch = np.concatenate([self.x_train[indices_spoof], self.live_train[indices_live]], axis=0)
        y_batch = np.concatenate([np.zeros(shape=(len(indices_spoof), 1)), np.zeros(shape=(len(indices_live),1))], axis=0)

        if np.random.uniform(0,1) < self.flip_prob:
            x_batch = self.flip(x_batch)
        if np.random.uniform(0,1) < self.mask_prob:
            x_batch = self.apply_grid_mask(x_batch)
        x_batch = preprocess_input(x_batch)

        return x_batch, y_batch
