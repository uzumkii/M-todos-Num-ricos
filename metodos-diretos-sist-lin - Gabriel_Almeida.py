import numpy as np

# Troca as linhas i e j da matriz A

def trocar_linhas(A,i,j):
    A[[i,j]] = A[[j,i]]

# Realiza o escalonamento da linha "linha" com o pivô "pivo"

def escalonar(pivo, linha, A):
    m = A[linha, pivo] / A[pivo, pivo]
    A[linha, :] = A[linha, :] - m * A[pivo, :]
    return m

# Implementa eliminação Gaussiana simples, onde A é uma matriz aumentada

def elim_gauss(A):
    n = len(A)
    for i in range(0, n):
        for k in range(i + 1, n):
            escalonar(i, k, A)

# Encontra a solução de um sistema triangular superior por substituição, onde A é uma matriz aumentada

def res_sistema_triangular_sup(A):
    n = len(A)
    x = [0] * n
    for i in reversed(range(n)):            #Laço "for" de n-1 até 0
        soma = 0
        for j in range(i+1,n):
            soma = soma + A[i,j] * x[j]
        x[i] = (A[i,n] - soma) / A[i,i]     #Lembre que a última coluna de A é b
    return np.array(x)

# Implementa eliminação Gaussiana com pivoteamento parcial

def elim_gauss_pivoteamento(A):
    n = len(A)
    for i in range(0, n):

        # Busca o índice do elemento de maior valor absoluto na coluna i (a partir da linha i)

        indice_max = i
        for k in range(i+1, n):
            if abs(A[k, i]) > abs(A[indice_max, i]):
                indice_max = k

        # Se o pivô não está na linha i, troca as linhas

        if indice_max != i:
            trocar_linhas(A, i, indice_max)

        # Elimina os elementos abaixo do pivô

        for k in range(i+1, n):
            if A[k, i] != 0:
                escalonar(i, k, A)

# Implementa o algoritmo de eliminação de Gauss-Jordan

def elim_gauss_jordan(A):
    n = len(A)
    for i in range(0, n):

        # Busca o índice do elemento de maior valor absoluto na coluna i (a partir da linha i)

        indice_max = i
        for k in range(i+1, n):
            if abs(A[k, i]) > abs(A[indice_max, i]):
                indice_max = k

        # Se o pivô não está na linha i, troca as linhas

        if indice_max != i:
            trocar_linhas(A, i, indice_max)

        # Elimina os elementos em todas as outras linhas (acima e abaixo)

        for k in range(n):
            if k != i and A[k, i] != 0:
                escalonar(i, k, A)


# Encontra a solução de um sistema linear diagonal (resultado da eliminação de Gauss-Jordan)

def res_sistema_diagonal(A):
    n = len(A)
    x = np.zeros(n)
    for i in range(n):

        # Divide o termo independente pelo elemento da diagonal

        x[i] = A[i, n] / A[i, i]
    return x


# Implementa o algoritmo de fatoração LU (sem pivoteamento)

def fatoracao_L_U(A):
    n = len(A)
    L = np.eye(n, dtype=float)
    U = A.copy().astype(float)
    for i in range(n):
        for k in range(i+1, n):
            if U[k, i] != 0:
                m = escalonar(i, k, U)
                L[k, i] = m
    return L, U

# Resolve o sistema LUx = b, onde L é triangular inferior e U triangular superior

def res_sistema_triangular_inf(A):

    # Substituição progressiva para sistema triangular inferior
    # A é a matriz aumentada [L | b]

    n = len(A)
    x = np.zeros(n)
    for i in range(n):
        soma = 0.0
        for j in range(i):
            soma = soma + A[i, j] * x[j]
        x[i] = (A[i, n] - soma) / A[i, i]
    return x

def res_sist_L_U(L, U, b):
    n = len(b)

    # Passo 1: substituição progressiva — resolve Ly = b

    b_col = b.reshape(-1, 1)
    L_aug = np.hstack((L, b_col))
    y = res_sistema_triangular_inf(L_aug)

    # Passo 2: substituição regressiva — resolve Ux = y

    y_col = y.reshape(-1, 1)
    U_aug = np.hstack((U, y_col))
    x = res_sistema_triangular_sup(U_aug)
    return x

#Cabeçalho do programa

print("\n"+ "Nomes: Gabriel Almeida; Gabriel Gimonski; Gustavo Lopes")
print("GRR: 20254589; 20252011; 20254578")
print("Curso: Eng Elétrica - UFPR")

# Teste do código

A = np.array([
    [10, 2, 1],
    [1, 5, 1],
    [2, 3, 10]
], dtype=float)

b = np.array([7, -8, 6], dtype=float)   # Termo independente

b_col = b.reshape(-1, 1)                # Transforma b em um vetor coluna
B = np.hstack((A, b_col))               # Constrói a matriz aumentada

print("\n" + "="*60)
print("Teste do codigo")
print("="*60 + "\n")

print("Matriz aumentada")
print(B)

print("\nResultado da eliminação Gaussiana:")
elim_gauss(B)
print(B)

print("\nSolução do sistema triangular superior: ")
x = res_sistema_triangular_sup(B)
print(x)

print("\nVerificando que Ax = b: ")
print(str(A @ x) + " =? " + str(b))     #o operador @ realiza multiplicação de matrizes (do tipo np.array)

# EXERCÍCIO 1 — Resolução do sistema 4x4 via Gauss-Jordan

print("\n" + "="*60)
print("EXERCÍCIO 1 — Eliminação de Gauss-Jordan")
print("="*60)

A1 = np.array([
    [-2, -4, -2, 2],
    [-4,  4, -4, 4],
    [ 1, -1,  2,-1],
    [-1,  3, -4, 4]
], dtype=float)
b1 = np.array([-4, 4, 2, -3], dtype=float)

b1_col = b1.reshape(-1, 1)
B1 = np.hstack((A1, b1_col))
print("Matriz aumentada inicial:")
print(B1)

elim_gauss_jordan(B1)
print("\nMatriz após Gauss-Jordan (diagonal):")
print(B1)

sol1 = res_sistema_diagonal(B1)
print("\nSolução (x1, x2, x3, x4):", sol1)
print("Verificação A @ x == b:", np.allclose(A1 @ sol1, b1))

# EXERCÍCIO 2 — Fatoração LU e resolução para múltiplos b

print("\n" + "="*60)
print("EXERCÍCIO 2 — Fatoração LU")
print("="*60)

A2 = np.array([
    [-2, -4, -2, 2],
    [-4,  4, -4, 4],
    [ 1, -1,  2,-1],
    [-1,  3, -4, 4]
], dtype=float)

L2, U2 = fatoracao_L_U(A2)
print("Matriz L:")
print(L2)
print("\nMatriz U:")
print(U2)
print("\nVerificação L @ U == A:", np.allclose(L2 @ U2, A2))

b_1 = np.array([-8, 8, 1, 9], dtype=float)
b_2 = np.array([ -4, 52, -16, 44], dtype=float)

x_b1 = res_sist_L_U(L2, U2, b_1)
print("\nSolução para b1 =", b_1, ":")
print("x =", x_b1)
print("Verificação A @ x == b1:", np.allclose(A2 @ x_b1, b_1))

x_b2 = res_sist_L_U(L2, U2, b_2)
print("\nSolução para b2 =", b_2, ":")
print("x =", x_b2)
print("Verificação A @ x == b2:", np.allclose(A2 @ x_b2, b_2))

# EXERCÍCIO 3 — Sistema 20x20 via eliminação com pivoteamento

print("\n" + "="*60)
print("EXERCÍCIO 3 — Sistema 20x20 (Gauss com pivoteamento parcial)")
print("="*60)

A3 = np.array([
    [ 3.0, 8.0, -1.0, 4.0, 2.0, -5.0, 1.0, 0.0, 6.0, -2.0, 3.0, 1.0, -4.0, 2.0, 7.0, -1.0, 0.0, 5.0, -3.0, 2.0],
    [ 5.0, -2.0, 7.0, 1.0, -4.0, 3.0, 8.0, -2.0, 0.0, 6.0, 1.0, -3.0, 5.0, 2.0, -1.0, 4.0, 7.0, 0.0, -5.0, 3.0],
    [-1.0, 6.0, 2.0, 5.0, -3.0, 7.0, 1.0, -4.0, 8.0, 2.0, 0.0, -5.0, 3.0, 6.0, -2.0, 1.0, -4.0, 8.0, 2.0, 0.0],
    [ 4.0, -3.0, 5.0, 0.0, 6.0, -2.0, 3.0, 1.0, -4.0, 7.0, -1.0, 2.0, 8.0, -5.0, 3.0, 6.0, -2.0, 1.0, 0.0, 4.0],
    [ 2.0, 7.0, -4.0, 8.0, 1.0, 5.0, -3.0, 6.0, -2.0, 0.0, 4.0, 7.0, -1.0, 3.0, 5.0, -2.0, 8.0, -4.0, 1.0, 6.0],
    [-5.0, 1.0, 3.0, -2.0, 6.0, -1.0, 4.0, 7.0, -1.0, 5.0, -3.0, 2.0, 0.0, 8.0, -4.0, 3.0, 6.0, -2.0, 7.0, -1.0],
    [ 1.0, -4.0, 8.0, 3.0, -5.0, 2.0, 0.0, 6.0, -2.0, 4.0, 7.0, -1.0, 5.0, -3.0, 2.0, 8.0, -4.0, 1.0, 6.0, -2.0],
    [ 0.0, 5.0, -2.0, 1.0, 4.0, 7.0, -1.0, 3.0, 5.0, -3.0, 2.0, 8.0, -4.0, 1.0, 6.0, -2.0, 0.0, 7.0, -1.0, 5.0],
    [ 6.0, -2.0, 0.0, -4.0, 7.0, -1.0, 5.0, -3.0, 1.0, 8.0, -4.0, 2.0, 6.0, -2.0, 3.0, 1.0, -5.0, 4.0, 8.0, -3.0],
    [-2.0, 0.0, 6.0, 7.0, -3.0, 5.0, 2.0, 8.0, -4.0, 2.0, 1.0, 6.0, -2.0, 4.0, 7.0, -1.0, 3.0, -5.0, 2.0, 0.0],
    [ 3.0, 8.0, -5.0, -1.0, 2.0, -3.0, 7.0, 1.0, -4.0, 6.0, -2.0, 5.0, 3.0, -1.0, 4.0, 7.0, -2.0, 0.0, 6.0, -4.0],
    [ 1.0, -3.0, 3.0, 2.0, 5.0, 2.0, -1.0, 4.0, 7.0, 1.0, 8.0, 0.0, -5.0, 3.0, 6.0, -2.0, 1.0, -4.0, 8.0, 2.0],
    [-4.0, 5.0, 6.0, 8.0, -1.0, 0.0, 5.0, -3.0, 2.0, -2.0, 3.0, -5.0, 1.0, 7.0, -1.0, 4.0, 2.0, 6.0, -2.0, 3.0],
    [ 2.0, 2.0, -2.0, -5.0, 3.0, 8.0, -3.0, 2.0, 6.0, 4.0, -1.0, 3.0, 7.0, -2.0, 0.0, 5.0, -3.0, 1.0, 4.0, 7.0],
    [ 7.0, -1.0, 1.0, 3.0, 5.0, -4.0, 2.0, 6.0, -2.0, 7.0, 4.0, 6.0, -1.0, 0.0, 3.0, -5.0, -2.0, 8.0, -4.0, 1.0],
    [-1.0, 4.0, -4.0, 6.0, -2.0, 3.0, 8.0, -2.0, 1.0, -1.0, 7.0, -2.0, 4.0, 5.0, -5.0, 0.0, 6.0, -2.0, 3.0, 8.0],
    [ 0.0, 7.0, 8.0, -2.0, 8.0, 6.0, -4.0, 0.0, -5.0, 3.0, -2.0, 1.0, 2.0, -3.0, 2.0, 6.0, -1.0, 4.0, 7.0, -1.0],
    [ 5.0, 0.0, 2.0, 1.0, -4.0, -2.0, 1.0, 7.0, 4.0, -5.0, 0.0, -4.0, 6.0, 1.0, 8.0, -2.0, 4.0, -3.0, 1.0, 5.0],
    [-3.0, -5.0, 0.0, 0.0, 1.0, 7.0, 6.0, -1.0, 8.0, 2.0, 6.0, 8.0, -2.0, 4.0, -4.0, 3.0, 7.0, 1.0, 2.0, -5.0],
    [ 2.0, 3.0, 4.0, 4.0, 6.0, -1.0, -2.0, 5.0, -3.0, 0.0, -4.0, 2.0, 3.0, 7.0, 1.0, 8.0, -1.0, 5.0, -5.0, 0.0]
], dtype=float)

b3 = np.array([12., -4., 18., 5., 22., -9., 3., 11., 0., -7., 14., 31., -2., 19., 8., -5., 10., 6., -12., 4.], dtype=float)

b3_col = b3.reshape(-1, 1)
B3 = np.hstack((A3, b3_col))
elim_gauss_pivoteamento(B3)
sol3 = res_sistema_triangular_sup(B3)
print("Solução x:")
for i, val in enumerate(sol3):
    print(f"  x[{i+1}] = {val:.6f}")
print("Verificação A @ x == b:", np.allclose(A3 @ sol3, b3))