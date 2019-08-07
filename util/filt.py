import numpy as np

def less_zero(arr,num = 7):
    index = np.linspace(0,len(arr)-1,len(arr),dtype='int')
    cnt = 0
    for i in range(2,len(arr)-2):
        if arr[i] != 0:
            arr[i] = arr[i]
            if cnt != 0:
                if cnt <= num*2:
                    arr[i-cnt:round(i-cnt/2)] = arr[i-cnt-1-2]
                    arr[round(i-cnt/2):i] = arr[i+2]
                    index[i-cnt:round(i-cnt/2)] = i-cnt-1-2
                    index[round(i-cnt/2):i] = i+2
                else:
                    arr[i-cnt:i-cnt+num] = arr[i-cnt-1-2]
                    arr[i-num:i] = arr[i+2] 
                    index[i-cnt:i-cnt+num] = i-cnt-1-2
                    index[i-num:i] = i+2
                cnt = 0
        else:
            cnt += 1
    return arr,index

def medfilt(data,window):
    if window%2 == 0 or window < 0:
        print('Error: the medfilt window must be even number')
        exit(0)
    pad = int((window-1)/2)
    pad_data = np.zeros(len(data)+window-1, dtype = type(data[0]))
    result = np.zeros(len(data),dtype = type(data[0]))
    pad_data[pad:pad+len(data)]=data[:]
    for i in range(len(data)):
        result[i] = np.median(pad_data[i:i+window])
    return result

def position_medfilt(positions,window):

    x,mask_index = less_zero(positions[:,0],window)
    y = less_zero(positions[:,1],window)[0]
    area = less_zero(positions[:,2],window)[0]
    x_filt = medfilt(x, window)
    y_filt = medfilt(y, window)
    area_filt = medfilt(area, window)
    cnt = 0
    for i in range(1,len(x)):
        if 0.8<x_filt[i]/(x[i]+1)<1.2 and 0.8<y_filt[i]/(y[i]+1)<1.2 and 0.6<area_filt[i]/(area[i]+1)<1.4:
            mask_index[i] = mask_index[i]
            if cnt != 0:
                mask_index[i-cnt:round(i-cnt/2)] = mask_index[i-cnt]
                mask_index[round(i-cnt/2):i] = mask_index[i] 
                cnt = 0
        else:
            mask_index[i] = mask_index[i-1]
            cnt += 1
    return mask_index

# def main():
#     import matplotlib.pyplot as plt
#     positions = np.load('../test_pos.npy')
#     positions_new = np.load('../test_pos.npy')
#     print(positions.shape)
#     mask_index = position_medfilt(positions.copy(), 7)
#     x = positions_new[2]

#     x_new = []
#     for i in range(len(x)):
#         x_new.append(x[mask_index[i]])

#     plt.subplot(211)
#     plt.plot(x)
#     plt.subplot(212)
#     plt.plot(x_new)
#     plt.show()

# if __name__ == '__main__':
#     main()