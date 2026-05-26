import numpy as np
import time

# Realiza a interpolação de Lagrange para calcular o valor de p(x) para o polinômio interpolador dos pares
# (x_dado[0], y_dado[0]), (x_dado[1], y_dado[2]), ..., (x_dado[n-1], y_dado[n-1])

def lagrange(x_dado, y_dado, x):
    n = len(x_dado)
    res = 0
    for k in range(n):
        res = res + y_dado[k] * L(x_dado, x, k)
    return res

# Computa o valor do coeficiente de Lagrange L_k para os dados x_dado, o ponto x e o índice k.
# L_k(x) = produto de (x - x_dado[j])/(x_dado[k] - x_dado[j]) para todo j diferente de k.

def L(x_dado, x, k):
    n = len(x_dado)
    resultado = 1.0
    for j in range(n):
        if j != k:
            resultado = resultado * (x - x_dado[j]) / (x_dado[k] - x_dado[j])
    return resultado

# Preenche a matriz F de diferenças divididas dados os pares (x_dado[0], y_dado[0]), (x_dado[1], y_dado[2]), ..., (x_dado[n-1], y_dado[n-1]).
# Por convenção, é construída uma matriz triangular inferior, e a diagonal principal possui os coeficientes do polinômio interpolador de Newton.

def difs_divididas(x_dado, y_dado):
    n = len(x_dado)
    F = np.zeros((n,n), dtype=float)
    F[:,0] = y_dado

    # A ordem desses fors garante que a matriz é preenchida coluna por coluna

    for j in range(1, n):
        for i in range (j, n):
            # Diferença dividida: [x_i, ..., x_{i-j}]
            F[i, j] = (F[i, j-1] - F[i-1, j-1]) / (x_dado[i] - x_dado[i-j])
    return F

# Avalia o polinômio de Newton p(x) com os valores de x_dado e os valores já pré-computados de F. Faz isso de maneira eficiente, usando a regra de Horner

def avaliar_newton(x_dado, F, x):
    n = len(x_dado)
    res = F[n - 1, n - 1]
    for i in reversed(range(n-1)):
        res = F[i, i] + (x - x_dado[i]) * res
    return res
#Cabeçalho do programa

print("\n"+ "Nome: Gabriel Almeida dos Santos")
print("GRR: 20254589")
print("Curso: Eng Elétrica - UFPR")

#Teste do código
print("\n" + "="*60)
print("Teste do codigo")
print("="*60 + "\n")

x_dado = (1,2,3,4,5)
y_dado = (2,4,9,18,10)

print(lagrange(x_dado, y_dado, 2.5))  #deve ser 5.546875 com os valores acima

F = difs_divididas(x_dado, y_dado)
print(avaliar_newton(x_dado, F, 2.5)) #deve ser 5.546875 com os valores acima

# EXERCÍCIO 8 — Interpolação de Lagrange: p(1) e p(3)

print("\n" + "="*60)
print("EXERCÍCIO 8 — Interpolação de Lagrange")
print("="*60)

x8 = np.array([0, 2, 4, 6], dtype=float)
y8 = np.array([4, 7, 5, 9], dtype=float)

p1 = lagrange(x8, y8, 1)
p3 = lagrange(x8, y8, 3)
print(f"p(1) = {p1} ")
print(f"p(3) = {p3} ")

# EXERCÍCIO 9 — Diferenças divididas e Newton

print("\n" + "="*60)
print("EXERCÍCIO 9 — Diferenças divididas e Newton")
print("="*60)

x9 = np.array([0, 1, 2, 3, 4], dtype=float)
y9 = np.array([1, 3, 7, 13, 21], dtype=float)

F9 = difs_divididas(x9, y9)
print("Tabela de diferenças divididas (F):")
print(F9)
print(f"\nCoeficientes do polinômio de Newton (diagonal de F):")
for i in range(len(x9)):
    print(f"  c[{i}] = F[{i},{i}] = {F9[i,i]}")

p25 = avaliar_newton(x9, F9, 2.5)
print(f"\np(2.5) = {p25}")

# EXERCÍCIO 10 — Comparação de tempo: Lagrange vs Newton

print("\n" + "="*60)
print("EXERCÍCIO 10 — Comparação de tempo: Lagrange vs Newton")
print("="*60)

x10 = np.array([
    -2., -1.95959596, -1.91919192, -1.87878788, -1.83838384, -1.7979798,
    -1.75757576, -1.71717172, -1.67676768, -1.63636364, -1.5959596, -1.55555556,
    -1.51515152, -1.47474747, -1.43434343, -1.39393939, -1.35353535, -1.31313131,
    -1.27272727, -1.23232323, -1.19191919, -1.15151515, -1.11111111, -1.07070707,
    -1.03030303, -0.98989899, -0.94949495, -0.90909091, -0.86868687, -0.82828283,
    -0.78787879, -0.74747475, -0.70707071, -0.66666667, -0.62626263, -0.58585859,
    -0.54545455, -0.50505051, -0.46464646, -0.42424242, -0.38383838, -0.34343434,
    -0.3030303, -0.26262626, -0.22222222, -0.18181818, -0.14141414, -0.1010101,
    -0.06060606, -0.02020202, 0.02020202, 0.06060606, 0.1010101, 0.14141414,
    0.18181818, 0.22222222, 0.26262626, 0.3030303, 0.34343434, 0.38383838,
    0.42424242, 0.46464646, 0.50505051, 0.54545455, 0.58585859, 0.62626263,
    0.66666667, 0.70707071, 0.74747475, 0.78787879, 0.82828283, 0.86868687,
    0.90909091, 0.94949495, 0.98989899, 1.03030303, 1.07070707, 1.11111111,
    1.15151515, 1.19191919, 1.23232323, 1.27272727, 1.31313131, 1.35353535,
    1.39393939, 1.43434343, 1.47474747, 1.51515152, 1.55555556, 1.5959596,
    1.63636364, 1.67676768, 1.71717172, 1.75757576, 1.7979798, 1.83838384,
    1.87878788, 1.91919192, 1.95959596, 2.
], dtype=float)

y10 = np.array([
    9.96409421e-03, 7.83641190e-03, 4.28250846e-03, -9.03719436e-04,
    -7.85950320e-03, -1.66201420e-02, -2.70853675e-02, -3.89881085e-02,
    -5.18692011e-02, -6.06194400e-02, -7.76904546e-02, -8.86854522e-02,
    -9.68203263e-02, -1.00769118e-01, -9.91863746e-02, -9.08068002e-02,
    -7.45603680e-02, -4.96962300e-02, -1.59066193e-02, 2.65597858e-02,
    7.68066257e-02, 1.33232490e-01, 1.93518242e-01, 2.54667767e-01,
    3.13107656e-01, 3.64847058e-01, 4.05693922e-01, 4.31518441e-01,
    4.38549169e-01, 4.23682539e-01, 3.84782887e-01, 3.20948104e-01,
    2.32715967e-01, 1.22188440e-01, -6.94437028e-03, -1.49492408e-01,
    -2.98969661e-01, -4.47908437e-01, -5.88268883e-01, -7.11919821e-01,
    -8.11159428e-01, -8.79239015e-01, -9.10850649e-01, -9.02540119e-01,
    -8.53010779e-01, -7.63290998e-01, -6.36747683e-01, -4.78939982e-01,
    -2.97319713e-01, -1.00797274e-01, 1.00797274e-01, 2.97319713e-01,
    4.78939982e-01, 6.36747683e-01, 7.63290998e-01, 8.53010779e-01,
    9.02540119e-01, 9.10850649e-01, 8.79239015e-01, 8.11159428e-01,
    7.11919821e-01, 5.88268883e-01, 4.47908437e-01, 2.98969661e-01,
    1.49492408e-01, 6.94437028e-03, -1.22188440e-01, -2.32715967e-01,
    -3.20948104e-01, -3.84782887e-01, -4.23682539e-01, -4.38549169e-01,
    -4.31518441e-01, -4.05693922e-01, -3.64847058e-01, -3.13107656e-01,
    -2.54667767e-01, -1.93518242e-01, -1.33232490e-01, -7.68066257e-02,
    -2.65597858e-02, 1.59066193e-02, 4.96962300e-02, 7.45603680e-02,
    9.08068002e-02, 9.91863746e-02, 1.00769118e-01, 9.68203263e-02,
    8.86854522e-02, 7.76904546e-02, 6.50619440e-02, 5.18692011e-02,
    3.89881085e-02, 2.70853675e-02, 1.66201420e-02, 7.85950320e-03,
    9.03719436e-04, -4.28250846e-03, -7.83641190e-03, -9.96409421e-03
], dtype=float)

# Tarefa 1: Lagrange para 5000 pontos

inicio = time.perf_counter()
for i in range(5000):
    _ = lagrange(x10, y10, i / 1000)
fim = time.perf_counter()
tempo_lagrange = fim - inicio
print(f"Tarefa 1 (Lagrange)  — Tempo total: {tempo_lagrange:.4f} segundos")

# Tarefa 2: Newton (pré-calcula F uma vez, avalia para 5000 pontos)

inicio = time.perf_counter()
F10 = difs_divididas(x10, y10)
for i in range(5000):
    _ = avaliar_newton(x10, F10, i / 1000)
fim = time.perf_counter()
tempo_newton = fim - inicio
print(f"Tarefa 2 (Newton)    — Tempo total: {tempo_newton:.4f} segundos")

razao = tempo_lagrange / tempo_newton if tempo_newton > 0 else float('inf')
print(f"\nLagrange foi {razao:.1f}x mais lento que Newton")
