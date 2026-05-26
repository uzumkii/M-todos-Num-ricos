import numpy as np

# Implementa o método de Gauss-Jacobi, e imprime todas as soluções intermediárias
# OBSERVAÇÃO: a função atualizar_jacobi(A,n,b,x_prev) ainda não foi implementada, e deve ser implementada por você

def gauss_jacobi(A, b, x_0, eps, rel):
    n = len(A)
    x_prev = x_0.copy()
    x_next = atualizar_jacobi(A, n, b, x_prev)
    if (rel):
        erro = erro_rel(x_next, x_prev) 
    else:
        erro = erro_abs(x_next, x_prev)
    while (erro > eps):
        print(x_next)
        x_prev = x_next.copy()
        x_next = atualizar_jacobi(A, n, b, x_prev)
        if (rel):
            erro = erro_rel(x_next, x_prev) 
        else:
            erro = erro_abs(x_next, x_prev)
    return x_next

# Implementa o método de Gauss-Seidel, e imprime todas as soluções intermediárias
# OBSERVAÇÃO: a função atualizar_seidel(A, n, b, x_prev) ainda não foi implementada, e deve ser implementada por você

def gauss_seidel(A, b, x_0, eps, rel):
    n = len(A)
    x_prev = x_0.copy()
    x_next = atualizar_seidel(A, n, b, x_prev)
    if (rel):
        erro = erro_rel(x_next, x_prev) 
    else:
        erro = erro_abs(x_next, x_prev)
    while (erro > eps):
        print(x_next)
        x_prev = x_next.copy()
        x_next = atualizar_seidel(A, n, b, x_prev)
        if (rel):
            erro = erro_rel(x_next, x_prev) 
        else:
            erro = erro_abs(x_next, x_prev)
    return x_next

# Calcula x_next a partir de x_prev pelo método de Gauss-Jacobi.
# No método de Jacobi, todos os valores de x_next dependem exclusivamente de x_prev.

def atualizar_jacobi(A, n, b, x_prev):
    x_next = np.zeros(n)
    for i in range(n):
        soma = 0.0
        for j in range(n):
            if j != i:
                soma = soma + A[i, j] * x_prev[j]
        x_next[i] = (b[i] - soma) / A[i, i]
    return x_next

# Calcula x_next a partir de x_prev pelo método de Gauss-Seidel.
# No método de Seidel, cada componente de x_next usa os valores já atualizados
# de x_next para j < i, e os valores de x_prev para j > i.

def atualizar_seidel(A, n, b, x_prev):
    x_next = x_prev.copy()
    for i in range(n):
        soma = 0.0
        for j in range(n):
            if j != i:
                soma = soma + A[i, j] * x_next[j]
        x_next[i] = (b[i] - soma) / A[i, i]
    return x_next

# Calcula o erro absoluto entre x_next e x_prev

def erro_abs(x_next, x_prev):
    max = float("-inf")
    for i in range(len(x_next)):
        erro = abs(x_next[i] - x_prev[i])
        if erro > max:
            max = erro
    return max

#Calcula o erro relativo entre x_next e x_prev

def erro_rel(x_next, x_prev):
    max = float("-inf")
    for i in range(len(x_next)):
        aux = abs(x_next[i])
        if aux > max:
            max = aux
    return erro_abs(x_next, x_prev)/max

# Versão com contagem de iterações (sem impressões intermediárias)

def gauss_jacobi_contar(A, b, x_0, eps, rel):
    n = len(A)
    x_prev = x_0.copy()
    x_next = atualizar_jacobi(A, n, b, x_prev)
    if rel:
        erro = erro_rel(x_next, x_prev)
    else:
        erro = erro_abs(x_next, x_prev)
    contagem = 1
    while erro > eps:
        x_prev = x_next.copy()
        x_next = atualizar_jacobi(A, n, b, x_prev)
        if rel:
            erro = erro_rel(x_next, x_prev)
        else:
            erro = erro_abs(x_next, x_prev)
        contagem += 1
    return x_next, contagem

# Versão com contagem de iterações (sem impressões intermediárias)

def gauss_seidel_contar(A, b, x_0, eps, rel):
    n = len(A)
    x_prev = x_0.copy()
    x_next = atualizar_seidel(A, n, b, x_prev)
    if rel:
        erro = erro_rel(x_next, x_prev)
    else:
        erro = erro_abs(x_next, x_prev)
    contagem = 1
    while erro > eps:
        x_prev = x_next.copy()
        x_next = atualizar_seidel(A, n, b, x_prev)
        if rel:
            erro = erro_rel(x_next, x_prev)
        else:
            erro = erro_abs(x_next, x_prev)
        contagem += 1
    return x_next, contagem

#cabeçalho

print("\n"+ "Nome: Gabriel Almeida dos Santos")
print("GRR: 20254589")
print("Curso: Eng Elétrica - UFPR")

#Teste do codigo

A = np.array([
    [10, 2, 1],
    [1, 5, 1],
    [2, 3, 10]
], dtype=float)

b = np.array([7, -8, 6], dtype=float)

print("=" * 50)
print("Teste do codigo")
print("=" * 50)
print("Matriz A:")
print(A)
print("Termo independente b:", b)

x_0 = np.zeros(3)
print("Solucao inicial:", x_0)
print()
print("--- Gauss-Jacobi | erro absoluto (eps=0.0001) ---")
print("Solucoes intermediarias:")
resultado = gauss_jacobi(A, b, x_0, 0.0001, False)
print("Solucao final:", resultado)
print()

print("--- Gauss-Jacobi | erro relativo (eps=0.0001) ---")
print("Solucoes intermediarias:")
resultado = gauss_jacobi(A, b, x_0, 0.0001, True)
print("Solucao final:", resultado)
print()

print("--- Gauss-Seidel | erro absoluto (eps=0.0001) ---")
print("Solucoes intermediarias:")
resultado = gauss_seidel(A, b, x_0, 0.0001, False)
print("Solucao final:", resultado)
print()

print("--- Gauss-Seidel | erro relativo (eps=0.0001) ---")
print("Solucoes intermediarias:")
resultado = gauss_seidel(A, b, x_0, 0.0001, True)
print("Solucao final:", resultado)

# EXERCÍCIO 5 — Sistema 3x3 com Gauss-Jacobi, eps=0.005

print("\n" + "="*60)
print("EXERCÍCIO 5 — Gauss-Jacobi com eps=0.005")
print("="*60)

A5 = np.array([
    [10, 1, 0],
    [1, 10, 1],
    [0, 1, 10]
], dtype=float)
b5 = np.array([11, 12, 11], dtype=float)
x0_5 = np.zeros(3)

sol5, iter5 = gauss_jacobi_contar(A5, b5, x0_5, 0.005, False)
print(f"Solução com Gauss-Jacobi (erro abs eps=0.005): {sol5}")
print(f"Número de iterações: {iter5}")

# EXERCÍCIO 6 — Sistema 20x20 com Jacobi e Seidel

print("\n" + "="*60)
print("EXERCÍCIO 6 — Sistema 20x20 (Jacobi e Seidel)")
print("="*60)

A6 = np.array([
    [ 50, -2, 1, 0, 3, -1, 2, 0, -1, 1, 0, 2, -3, 1, 0, -2, 1, 0, 1, -1],
    [ -2, 55, -3, 1, 0, 2, -1, 3, 0, -2, 1, 0, 1, -1, 2, 0, -1, 3, 0, 1],
    [ 1, -3, 48, -2, 1, 0, 3, -1, 2, 0, -1, 1, 0, 2, -2, 1, 0, 1, -3, 0],
    [ 0, 1, -2, 52, -4, 2, 0, 1, -3, 1, 0, -1, 2, 0, 3, -1, 2, 0, -1, 2],
    [ 3, 0, 1, -4, 60, -3, 1, 0, 2, -2, 1, 0, -1, 3, 0, 1, -2, 1, 0, -1],
    [ -1, 2, 0, 2, -3, 45, -2, 1, 0, 3, -1, 2, 0, -1, 1, 0, 2, -3, 1, 0],
    [ 2, -1, 3, 0, 1, -2, 58, -3, 1, 0, 2, -1, 3, 0, -1, 2, 0, 1, -2, 1],
    [ 0, 3, -1, 1, 0, 1, -3, 51, -4, 2, 0, 1, -2, 1, 0, -1, 3, 0, 2, -2],
    [ -1, 0, 2, -3, 2, 0, 1, -4, 49, -2, 1, 0, 3, -1, 2, 0, -1, 1, 0, 3],
    [ 1, -2, 0, 1, -2, 3, 0, 2, -2, 53, -4, 1, 0, 2, -1, 3, 0, -2, 1, 0],
    [ 0, 1, -1, 0, 1, -1, 2, 0, 1, -4, 56, -3, 2, 0, 1, -2, 1, 0, 3, -1],
    [ 2, 0, 1, -1, 0, 2, -1, 1, 0, 1, -3, 47, -4, 2, 0, 1, -2, 1, 0, 2],
    [ -3, 1, 0, 2, -1, 0, 3, -2, 3, 0, 2, -4, 62, -3, 1, 0, 2, -1, 1, 0],
    [ 1, -1, 2, 0, 3, -1, 0, 1, -1, 2, 0, 2, -3, 50, -2, 1, 0, 3, -1, 2],
    [ 0, 2, -2, 3, 0, 1, -1, 0, 2, -1, 1, 0, 1, -2, 54, -3, 2, 0, 1, -1],
    [ -2, 0, 1, -1, 1, 0, 2, -1, 0, 3, -2, 1, 0, 1, -3, 46, -4, 2, 0, 1],
    [ 1, -1, 0, 2, -2, 2, 0, 3, -1, 0, 1, -2, 2, 0, 2, -4, 57, -3, 1, 0],
    [ 0, 3, 1, 0, 1, -3, 1, 0, 1, -2, 0, 1, -1, 3, 0, 2, -3, 59, -2, 1],
    [ 1, 0, -3, -1, 0, 1, -2, 2, 0, 1, 3, 0, 1, -1, 1, 0, 1, -2, 44, -4],
    [ -1, 1, 0, 2, -1, 0, 1, -2, 3, 0, -1, 2, 0, 2, -1, 1, 0, 1, -4, 52]
], dtype=float)

b6 = np.array([
    120, -85, 45, 110, -30, 75, 160, -20, 95, 130,
    -40, 80, 145, -60, 105, 35, 115, -15, 70, 125
], dtype=float)

x0_6 = np.zeros(20)
eps6 = 1e-6

print("\n--- Erro Absoluto (eps = 1e-6) ---")
sol_jac_abs, iter_jac_abs = gauss_jacobi_contar(A6, b6, x0_6, eps6, False)
print(f"Gauss-Jacobi  — iterações: {iter_jac_abs}")
sol_sei_abs, iter_sei_abs = gauss_seidel_contar(A6, b6, x0_6, eps6, False)
print(f"Gauss-Seidel  — iterações: {iter_sei_abs}")

print("\n--- Erro Relativo (eps = 1e-6) ---")
sol_jac_rel, iter_jac_rel = gauss_jacobi_contar(A6, b6, x0_6, eps6, True)
print(f"Gauss-Jacobi  — iterações: {iter_jac_rel}")
sol_sei_rel, iter_sei_rel = gauss_seidel_contar(A6, b6, x0_6, eps6, True)
print(f"Gauss-Seidel  — iterações: {iter_sei_rel}")

print("\nSolução Jacobi (abs):", sol_jac_abs)
print("Verificação A @ x == b:", np.allclose(A6 @ sol_jac_abs, b6, atol=1e-4))
