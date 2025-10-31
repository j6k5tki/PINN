# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 00:12:44 2025

@author: j6k5tki
"""
import numpy as np
import matplotlib.pyplot as plt
# t0: 실험 데이터의 시간 배열 (Z에 해당)
# T : 재구성 데이터의 시간 배열 (N에 해당)
# N : 재구성된 impurity density, shape (nt, nr) 또는 (50,50)
# Z : 실험 impurity density, shape (nt, nr) 또는 (50,70)

r0 = np.load('./26861/radius.npy')
t0 = np.load('./26861/time.npy')
N0 = np.load('./26861/imp1d.npy')/1e17 # (50,70)

R0, T0 = np.meshgrid(r0, t0)            # (50,70) 형태의 좌표

pc1 = plt.pcolormesh(R0,T0,N0.reshape(71,50),cmap='jet',vmax=4)
plt.xlim([0,0.8])
plt.ylim([3.2,6])
plt.show()

X = np.load('./inverse_X_202503202235.npy')
Y = np.load('./inverse_Y_202503202235.npy')

R = X[:,0].reshape(50,50) / 0.43
T = X[:,1].reshape(50,50) + 3
N = Y[:,0].reshape(50,50)
D = Y[:,1].reshape(50,50)
V = Y[:,2].reshape(50,50)

pc2 = plt.pcolormesh(R,T,N,cmap='jet',vmax=4)
plt.xlim([0,0.8])
plt.ylim([3.2,6])
plt.show()

import numpy as np

# ----- 입력 -----
# r0 : (nr,)  예: 50
# t0 : (nt0,) 예: 71, 실험 데이터 시간 (N0)
# N0 : (nt0, nr)  실험 impurity density (예: 71x50)
# T  : (nt, ) 또는 (nt, nr) 재구성 데이터 시간 (N)
# N  : (nt, nr)  재구성 impurity density (예: 50x50)

# 0) 시간 벡터 정리
t_exp = np.ravel(t0)                              # (nt0,)
# T가 meshgrid면 한 열만 사용해 1D 시간 벡터 생성
t_rec = T[:, 0] if (hasattr(T, "ndim") and T.ndim == 2) else np.ravel(T)  # (nt,)

# 1) N0의 축 확인/정렬: (time, radius) 형태가 아니면 전치
if N0.shape[0] != t_exp.size and N0.shape[1] == t_exp.size:
    N0 = N0.T  # -> (nt0, nr)

# 시간 배열이 단조 증가가 아니면 정렬
if np.any(np.diff(t_exp) < 0):
    order = np.argsort(t_exp)
    t_exp = t_exp[order]
    N0 = N0[order, :]
if np.any(np.diff(t_rec) < 0):
    order = np.argsort(t_rec)
    t_rec = t_rec[order]
    N = N[order, :]

# 2) 실험 N0을 재구성 시간축(t_rec)에 보간
N0_interp = np.empty_like(N)                       # (nt, nr)
for ir in range(N0.shape[1]):
    N0_interp[:, ir] = np.interp(t_rec, t_exp, N0[:, ir])

# 3) 시간 구간 선택 (3.2–6.0 s)
tmin, tmax = 4, 6.0
mask = (t_rec >= tmin) & (t_rec <= tmax)
N_cut   = N[mask, :]
N0_cut  = N0_interp[mask, :]

# 4) 편차 지표 계산
#    - MAPE(평균 상대 편차, %)  : mean(|N-N0| / |N0|)
#    - RMSE                      : sqrt(mean((N-N0)^2))
#    - 상대 L2 오차(%)           : ||N-N0||_2 / ||N0||_2 * 100
#    - r별 상대 편차 프로파일(%)

eps = 1e-12  # 0분모 방지용
diff = N_cut - N0_cut
abs_diff = np.abs(diff)

den = np.maximum(np.abs(N0_cut), eps)
mape_pct = float(np.mean(abs_diff / den) * 100.0)

#plt.pcolormesh((abs_diff / den) * 100.0)
#plt.show()

mse  = float(np.mean(diff**2))
rmse = float(np.sqrt(mse))

rel_L2_pct = float(
    100.0 * np.linalg.norm(diff.ravel(), 2) / max(np.linalg.norm(N0_cut.ravel(), 2), eps)
)

# 반경(r)별 평균 상대 편차(%)
rel_by_r_pct = np.mean(abs_diff / den, axis=0) * 100.0  # shape: (nr,)

print("=== Deviation metrics (t = 3.2–6.0 s) ===")
print(f"MAPE (mean relative deviation): {mape_pct:.3f} %")
print(f"RMSE: {rmse:.3e}")
print(f"Relative L2 error: {rel_L2_pct:.3f} %")
print(f"rel_by_r_pct shape: {rel_by_r_pct.shape}")

# ----- r/a < 0.9 마스크 -----
ra_mask = r0 < 0.9

# 반경(r)별 평균 상대 편차(%)
rel_by_r_pct = np.mean(abs_diff / den, axis=0) * 100.0  # shape: (nr,)

# r/a < 0.9 구간의 평균 상대 편차(%)
mape_core_pct = float(np.mean(rel_by_r_pct[ra_mask]))

print("=== Deviation metrics (t = 3.2–6.0 s, r/a < 0.9) ===")
print(f"MAPE_core (mean relative deviation, r/a<0.9): {mape_core_pct:.3f} %")
print(f"RMSE (all r): {rmse:.3e}")
print(f"Relative L2 error (all r): {rel_L2_pct:.3f} %")

# (선택) r/a별 상대 편차 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(5.2,3.6))
plt.plot(r0, rel_by_r_pct, lw=2)
plt.xlabel("r/a"); plt.ylabel("Mean relative deviation (%)")
plt.title("Reconstructed vs Experimental (3.2–6.0 s)")
plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()


