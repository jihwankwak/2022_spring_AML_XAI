import  os.path
import  numpy as np


class SineNShot:

    def __init__(self, batchsz, k_shot, k_query):
        """
        Different from mnistNShot, the
        :param batchsz: task num
        :param k_shot:
        :param k_qry:
        """

        self.batchsz = batchsz
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}

        self.datasets_cache = {"train": self.load_data_cache(),  # current epoch data cached
                               "test": self.load_data_cache()}


    def load_data_cache(self):
        """
        Collects several batches data for N-shot learning
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot
        querysz = self.k_query
        data_cache = []


        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set

                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                
                amplitude = np.random.uniform(0.1, 5)
                phase = np.random.uniform(0, np.pi)
                
                x_spt = np.random.rand(setsz) * 10 - 5
                x_qry = np.random.rand(querysz) * 10 - 5
                y_spt = np.sin(x_spt + phase) * amplitude
                y_qry = np.sin(x_qry + phase) * amplitude
                

                # append [sptsz, 1] => [b, setsz, 1]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)


            # [b, setsz, 1]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, 1)
            y_spts = np.array(y_spts).astype(np.float32).reshape(self.batchsz, setsz, 1)
            # [b, qrysz]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, 1)
            y_qrys = np.array(y_qrys).astype(np.float32).reshape(self.batchsz, querysz, 1)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache()

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch

