import numpy as np

class Smoothing:
    def __init__(self, seq_len, vector_len, smoothing_radius, output_index):
        super(Smoothing, self).__init__()
        if smoothing_radius > output_index or smoothing_radius > seq_len-(output_index+1):
            raise ValueError(f'Smoothing range is too big. max value is {min(output_index,seq_len-(output_index+1))}')

        self.avseq_len = seq_len - output_index + smoothing_radius
        self.smooth_len = 2 * smoothing_radius + 1

        self.avseq = None
        # buat pembagi (frame wise)
        self.div = np.empty((self.avseq_len-1,vector_len))
        for i in range(self.avseq_len-1):
            for j in range(vector_len):
                self.div[i,j] = self.avseq_len-i

        self.x_sm_i = output_index-smoothing_radius
    def __call__(self, x):
        if self.avseq is None:
            self.avseq =  x[-self.avseq_len:]            
        else:
            # average iteratif masukin di index sebelumnya
            self.avseq[:-1] = self.avseq[1:] + (x[self.x_sm_i:-1]-self.avseq[1:])/self.div
            # index terakhir copy aja
            self.avseq[-1] = x[-1]
        return self.avseq[0:self.smooth_len].mean(axis=0)
        
    def reset(self):
        self.avseq=None


##### TEST #####
# import time
# n = np.zeros((7, 4))
# for i in range(7):
#     for j in range(4):
#         n[i][j] = i
# smoothing = Smoothing(7, 4, 1, 3)
# print(n)

# start = time.time()
# for i in range(5):
#     x= smoothing(n)
#     print(i + 1, x)
# print("time",time.time()-start)