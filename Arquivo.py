import numpy as np
import random
import sys
from math import sqrt

# Dados das cidades
x = np.array([0.77687122244663642, 0.5572653296455039, 0.65639441309858648,
              0.60439895238077324, 0.10984792404443977, 0.30681838758814639,
              0.036420458719028548, 0.50750194272285054, 0.79819787712259027,
              0.79896874846157562, 0.14326939769923641, 0.071101926660729675,
              0.72613149506352259, 0.22624105387667293, 0.6248041238023041,
              0.5483227916626594, 0.39699387912590556, 0.075454958741316913,
              0.67595096782693853, 0.074297051769727118, 0.77687122244663642])

y = np.array([0.27943919986108079, 0.11661366329340583, 0.39053913424199571,
              0.66616903964750607, 0.6985758378186272, 0.20730006383213373,
              0.5024721283845478, 0.073938685056537334, 0.67991802460956252,
              0.39749277989717913, 0.14151256215331487, 0.12773617026441342,
              0.37197289724774407, 0.69033435138929333, 0.9189034809361033,
              0.52333815217506263, 0.42525694545543524, 0.37166915101708831,
              0.99033329254439939, 0.15694231625653665, 0.27943919986108079])


def cv_fitness(individuos, x, y):
    """
    Função de aptidão para o problema do caixeiro viajante

        individuos: matriz 20x20 onde cada linha é uma rota possível
        x: array com coordenadas x das cidades
        y: array com coordenadas y das cidades

    Retorna:
        matriz com as distâncias totais de cada rota
    """
    num_individuos, num_cidades = individuos.shape
    tour = np.hstack((individuos, individuos[:, 0:1]))  # Fecha o ciclo

    # Calcula matriz de distâncias entre todas as cidades
    dist_matrix = np.zeros((num_cidades, num_cidades))
    for i in range(num_cidades):
        for j in range(num_cidades):
            dist_matrix[i, j] = sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)

    # Calcula a distância total para cada rota
    distancias = np.zeros(num_individuos)
    for i in range(num_individuos):
        for j in range(num_cidades):
            cidade_atual = int(tour[i, j]) - 1  # Convertendo para índice 0-based
            cidade_proxima = int(tour[i, j + 1]) - 1
            distancias[i] += dist_matrix[cidade_atual, cidade_proxima]

    # Obtém os índices que ordenariam as distâncias em ordem crescente
    indices_ordenados = np.argsort(distancias)

    return distancias[indices_ordenados], tour[indices_ordenados]


def armazena_dez_melhores(distancias, individuos):
    # Seleciona os 10 melhores
    primeiros_10_individuos = individuos[:10, :-1].astype(int)
    primeiras_10_distancias = distancias[:10]
    return primeiras_10_distancias, primeiros_10_individuos


def fazer_crossover(pai1, pai2):
    # cria arrays de filhos
    tamanho = len(pai1)
    filho1 = np.full(tamanho, -1, dtype=int)
    filho2 = np.full(tamanho, -1, dtype=int)

    ciclo = 0
    usados = set()

    while len(usados) < tamanho:
        # encontra um índice não usado
        for i in range(tamanho):
            if i not in usados:
                inicio = i
                break

        indice = inicio
        while True:
            # copia gene segundo o ciclo
            if ciclo % 2 == 0:
                filho1[indice], filho2[indice] = pai1[indice], pai2[indice]
            else:
                filho1[indice], filho2[indice] = pai2[indice], pai1[indice]

            usados.add(indice)

            # próximo valor a buscar em pai1
            proximo_valor = pai2[indice]
            pos = np.where(pai1 == proximo_valor)[0]
            if pos.size == 0:
                break

            indice = pos[0]
            if indice == inicio:
                break

        ciclo += 1

    return filho1, filho2

def mutacao_filhos(filho1, filho2):
    filho1_mutado = np.copy(filho1)
    filho2_mutado = np.copy(filho2)

    operador1 = random.randint(0, len(filho1) - 1)
    operador2 = random.randint(0, len(filho1) - 1)

    # se sortear o mesmo operador, repete
    if operador1 == operador2:
        return mutacao_filhos(filho1, filho2)

    filho1_mutado[operador1], filho1_mutado[operador2] = filho1_mutado[operador2], filho1_mutado[operador1]
    filho2_mutado[operador1], filho2_mutado[operador2] = filho2_mutado[operador2], filho2_mutado[operador1]

    return filho1_mutado, filho2_mutado


def gerar_filhos(populacao_pais):
    # Roleta proporcional aos top 10 (peso decrescente)
    roleta = []
    for i, pai in enumerate(populacao_pais):
        roleta.extend([pai] * (10 - i))

    filhos = []
    for _ in range(5):
        # pga dois pais aleatorios e gera dois filhos fazendo o crossover e mutacao
        pai1 = random.choice(roleta)
        pai2 = random.choice(roleta)
        filho1, filho2 = fazer_crossover(pai1, pai2)
        filho1, filho2 = mutacao_filhos(filho1, filho2)
        filhos.append(filho1)
        filhos.append(filho2)

    return np.array(filhos, dtype=int)


if __name__ == "__main__":
    num_cidades = len(x)
    num_individuos = 20
    num_iteracoes = 1000

    # Gera população inicial (cada linha é uma permutação das cidades)
    # +1 porque as cidades são numeradas de 1 a N
    population = np.array([np.random.permutation(num_cidades) +1 for _ in range(num_individuos)])
 
    for iteracao in range(num_iteracoes):
        # Calcula as distancias, ja adiciona o puxadinho nos individuos e ja vem ordenado 
        distancia, individuos = cv_fitness(population, x, y)

        # retorna apenas os 10 melhores
        menores_distancias, melhores_individuos = armazena_dez_melhores(distancia, individuos)

        print(f"Geração {iteracao}: Melhores distâncias = {menores_distancias}\n")

        # Gera novos filhos e monta nova população
        filhos = gerar_filhos(melhores_individuos)
        population = np.vstack((melhores_individuos, filhos))
