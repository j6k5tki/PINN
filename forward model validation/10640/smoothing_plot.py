# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 20:55:02 2025

@author: j6k5tki
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def load_data(filename):
    """
    DV_data_106401.txt 파일에서 r/a, D, V 데이터를 읽어 리스트(또는 NumPy 배열)로 반환합니다.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # r/a, D, V 데이터가 각각 "#r/a", "#D", "#V" 라벨 뒤줄에 있으므로 이를 검색
    # 예) #r/a 뒤줄이 r값, #D 뒤줄이 D값, #V 뒤줄이 V값
    # (참고: 파일 구조에 따라 달라질 수 있으니, 실제 파일 구조와 일치하는지 확인 필요)
    
    # 라벨 위치 탐색
    idx_r = None
    idx_d = None
    idx_v = None
    
    for i, line in enumerate(lines):
        # 라벨을 찾아서 그 다음줄이 실제 데이터
        if line.strip().lower().startswith('#r/a'):
            idx_r = i + 1
        elif line.strip().lower().startswith('#d'):
            idx_d = i + 1
        elif line.strip().lower().startswith('#v'):
            idx_v = i + 1
    
    # r/a
    r_line = lines[idx_r].strip()
    r_vals = [float(x) for x in r_line.split()]
    
    # D
    d_line = lines[idx_d].strip()
    d_vals = [float(x) for x in d_line.split()]
    
    # V
    v_line = lines[idx_v].strip()
    v_vals = [float(x) for x in v_line.split()]
    
    return np.array(r_vals), np.array(d_vals), np.array(v_vals)

def smooth_profiles(r, D, V, window_length=5, polyorder=2):
    """
    Savitzky-Golay 필터를 사용해 D, V 프로파일을 스무딩합니다.
    window_length와 polyorder는 필터 파라미터로,
    window_length는 반드시 데이터 길이보다 짧은 홀수여야 하고,
    polyorder < window_length 여야 합니다.
    """
    D_smooth = savgol_filter(D, window_length, polyorder)
    V_smooth = savgol_filter(V, window_length, polyorder)
    return D_smooth, V_smooth

def smooth(a,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

def main():
    # 1) 데이터 로드
    filename = 'DV_data_106401.txt'  # 동일 디렉터리에 있다고 가정
    r, D, V = load_data(filename)
    
    # 2) 스무딩
    D_smooth, V_smooth = smooth_profiles(r, D, V, window_length=5, polyorder=2)
    
    D_smooth = smooth(D, 3)
    V_smooth = smooth(V, 3)
    
    # 3) 결과 시각화
    plt.figure(figsize=(8,5))

    plt.subplot(1,2,1)
    plt.plot(r, D, 'r--', label='Original D')
    plt.plot(r, D_smooth, 'b-', label='Smoothed D')
    plt.xlabel('r/a')
    plt.ylabel('D')
    plt.legend()
    plt.title('D Profile')
    
    plt.subplot(1,2,2)
    plt.plot(r, V, 'r--', label='Original V')
    plt.plot(r, V_smooth, 'b-', label='Smoothed V')
    plt.xlabel('r/a')
    plt.ylabel('V')
    plt.legend()
    plt.title('V Profile')

    plt.tight_layout()
    plt.show()
    
    with open('DV_data_1064011.txt','w') as f:
        f.write('#10640 4750 24\n')
        f.write('#r/a\n')
        for i,R in enumerate(r):
            if i != len(r)-1:
                f.write('%.2f\t'%R)
            else:
                f.write('%.2f\n'%R)
        f.write('#D\n')
        for i,d in enumerate(D_smooth):
            if i != len(D_smooth)-1:
                f.write('%.2f\t'%d)
            else:
                f.write('%.2f\n'%d)
        f.write('#V\n')
        for i,v in enumerate(V_smooth):
            if i != len(V_smooth)-1:
                f.write('%.2f\t'%v)
            else:
                f.write('%.2f\n'%v)

if __name__ == "__main__":
    main()
