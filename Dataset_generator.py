from tensorflow.keras.utils import Sequence 
from tensorflow.keras.applications.mobilenet import preprocess_input


class DS_generator(Sequence):

    def __init__(self,
                 x_train,
                 y_train,
                 shuffle = True,
                 batch_size = 256):
        
        self.x_train = x_train
        self.y_train = y_train
        self.aug_prob = 0
        self.batch_size = batch_size
        self.shuffle = shuffle
        # print(type(self.x_train))


    def grid_mask(self, side, n):
        sq_0 = np.zeros((side, side, 3))
        sq_1 = np.full((side, side, 3), fill_value=1)
        row_0 = np.hstack([sq_0, sq_1] * n)
        row_1 = np.hstack([sq_1, sq_0] * n)
        mask = np.vstack([row_0, row_1] * n)
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


    def flip(self, batch):
        return batch[:,:,::-1,:]


    def __len__(self):                                      # возвращает количество бачей в датасете
        return len(self.x_train) // self.batch_size


    def __getitem__(self, idx):                                         # этот метод формирует батч
        if self.shuffle == True:
            indices = np.random.choice(len(self.x_train), size=self.batch_size)
        else:
            indices = np.arange(idx * self.batch_size, (idx + 1) * self.batch_size)
        
        indices = np.random.choice(len(self.x_train), size=self.batch_size)         # случайные индексы для бача (idx - номер бача в эпохе(?) - не используется)
        
        x_batch = self.x_train[indices]
        y_batch = self.y_train[indices]

        if np.random.normal() < 0.5:
            x_batch = self.flip(x_batch)
        if np.random.normal() < 0.3:
            x_batch = self.apply_grid_mask(x_batch)
        x_batch = preprocess_input(x_batch)

        return x_batch, y_batch
