import numpy as np
from numpy import round
print("-" * 50)
def calculate(x, isRound=False, dp=4):
    return round(x,dp) if isRound else x
def LSMAmericanPut(S, dt, r, K):
   M,N=S.shape
   C = np.maximum(K - S, 0)
   print("Intrinsic Values Matrix C:")
   print(np.array2string(C, formatter={'float_kind':lambda x: f"{x:6.2f}"}))
   
   #C_(i,j) -> C[j-1, i-1]
   p=np.ones(M, dtype=int)*N
   print()
   print(f"p: {p}")
   print("-" * 50)

   #p_j -> p[j-1]
   for i in range(N-1, 1-1, -1):
      x=[S[j-1,i-1] for j in range(1, M+1) if C[j-1,i-1]>0]
      print(f"At i = {i}: ")
      print()
      print(f"x: {x}")
      y=[calculate(np.exp(-r*(p[j-1]-i)*dt)*C[j-1, p[j-1]-1],isRound=True,dp=6) 
              for j in range(1, M+1) if C[j-1, i-1]>0]
      print(f"y: {y}")
      c=np.polyfit(x,y,2).round(4)
      print()
      print(f"f(x) = {c[0]}x^2 + {c[1]}x + {c[2]} ")
      print()

      for j in range(1, M+1):
         if C[j-1,i-1]>0:
            fS=np.polyval(c, S[j-1,i-1]).round(4)
            c_value = round(C[j-1][i-1], 5)
            print(f'Path {j}: {c_value} ? {fS} ' + str(c_value>fS))
            if C[j-1,i-1]>fS:
               p[j-1]=i
         else:
            print(f'Path {j}: -------------')

      print()
      print(f'Updated p: {p}')
      print("-" * 50)
   s = 0
   print("Terms in summation for V calculation:")
   for j in range(1, M + 1):
      discount_factor = np.exp(-r * p[j - 1] * dt)
      payoff = C[j - 1, p[j - 1] - 1]
      term = discount_factor * payoff
      print(f"Path {j}: {payoff:.2f} * exp(-{r} * {p[j - 1]} * {dt:.2f}) = {term:.4f}")
      s += term
   V = s / M
   return V

K, r, T, N=1.1, 0.06, 3, 3
dt=T/N
S=np.array([[1.09, 1.08, 1.34],
            [1.16, 1.26, 1.54],
            [1.22, 1.07, 1.03],
            [0.93, 0.97, 0.92],
            [1.11, 1.56, 1.52],
            [0.76, 0.77, 0.90],
            [0.92, 0.84, 1.01],
            [0.88, 1.22, 1.34]])
V=LSMAmericanPut(S, dt, r, K)
print()
print(round(V, 3))