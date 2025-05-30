import numpy as np
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

    # Ordena tanto as distâncias quanto a matriz tour de acordo com os índices
    distancias_ordenadas = distancias[indices_ordenados]
    tour_ordenado = tour[indices_ordenados]

    return distancias_ordenadas, tour_ordenado

def armazena_dez_melhores(distancias, individuos):
    # Seleciona os 10 melhores
    primeiras_10_distancias = distancias[:10]
    primeiras_10_individuos = individuos[:10]

    print(f" 10 melhores distâncias: {primeiras_10_distancias}")
    print(f" 10 pais melhores: {primeiras_10_individuos}")

    return distancias,individuos


def transformar_vetor(vetor):
    vetor_transformado = np.empty(len(vetor), dtype=object)  # Usando dtype=object para números grandes

    for i in range(len(vetor)):
        valor = vetor[i]
        if i == 0:
            novo_valor = valor  # Mantém o primeiro elemento sem alteração
        else:
            ultimo_digito = str(valor)[-1]  # Pega o último dígito
            novo_valor = int(str(valor) + ultimo_digito * i)  # Repete o último dígito 'i' vezes
        vetor_transformado[i] = novo_valor

    return vetor_transformado

# Exemplo de uso
if __name__ == "__main__":
    num_cidades = len(x)
    num_individuos = 20

    # Gera população inicial (cada linha é uma permutação das cidades)
    # +1 porque as cidades são numeradas de 1 a N
    population = np.array([np.random.permutation(num_cidades) +1 for _ in range(num_individuos)])

    # Faz o puxadinho da primeira coluna pra última
    for i in range(num_individuos):
        population[i, 20] = population[i, 0]

    # Calcula as distâncias
    distancias, individuos = cv_fitness(population, x, y)

    # Armazena os dez melhores, descarta os 10 piores
    distancias, individuos = armazena_dez_melhores(distancias, individuos)

    matriz_transformada = transformar_vetor(individuos)

    print(f" {matriz_transformada}")