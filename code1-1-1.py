import numpy as np
import scipy.linalg
import time

# 행렬 크기 설정
n = 1000
repeat = 100  # 반복 횟수 설정

# 난수 생성
np.random.seed(0)
A = np.random.rand(n, n)
B = [np.random.rand(n) for _ in range(repeat)]  # b 벡터 100개 생성

# 방법 1: 매번 LU 분해를 반복하며 선형계를 푸는 방식 (np.linalg.solve)
solve_times = []
for b in B:
    start_time = time.time()
    x = np.linalg.solve(A, b)  # 매번 LU 분해를 수행한 후 풀이
    solve_times.append(time.time() - start_time)

avg_time_solve = np.mean(solve_times)
print(f"매번 LU 분해 후 풀이 평균 시간 ({repeat}회): {avg_time_solve:.6f} 초")

# 방법 2: LU 분해를 최초 1회만 수행하고, 이후 결과만 이용해 빠르게 푸는 방식
start_time = time.time()
LU, piv = scipy.linalg.lu_factor(A)  # 최초 1회 LU 분해 수행
factor_time = time.time() - start_time

solve_with_lu_times = []
for b in B:
    start_time = time.time()
    x = scipy.linalg.lu_solve((LU, piv), b)  # 미리 분해된 LU를 이용해 빠르게 풀이
    solve_with_lu_times.append(time.time() - start_time)

avg_time_solve_with_lu = np.mean(solve_with_lu_times)
print(f"최초 1회 LU 분해 후 빠른 풀이 평균 시간 ({repeat}회): {avg_time_solve_with_lu:.6f} 초")
