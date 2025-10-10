import tensorflow as tf

class Normalizacao:
    def __init__(self, X):
        # Inicializa os atributos
        self.desvios_padroes = None
        self.medias          = None
        self.x = X
        self.size = self.x.shape[1]

    # Função para calcular a média de cada coluna
    def mean_(self):
        # Calcula a média de cada coluna
        self.medias = [ tf.round(tf.reduce_mean(self.x[:, i]), 2) for i in range(self.size)]

        # Retorna o próprio objeto
        return self

    def std_(self):
        # Calcula a média
        self.mean_()

        # Calcula o desvio padrão de cada coluna
        self.desvios_padroes = [
            tf.math.sqrt(tf.reduce_mean( (self.x[:, i] - self.medias[i])**2) ) for i in range(self.size)
        ]
        return self

    def transformacao(self):
        # Chamando funções para calcular média e desvio padrão
        self.std_()
        # Aplica a normalização z-score
        self.x = (self.x - self.medias) / self.desvios_padroes
        return self.x

    def inversa_transformacao(self):
        try:
            return tf.multiply(self.desvios_padroes, self.x)  + self.medias
        except Exception as e:
            print("talvez você não normalizou os dados ainda")
        return self.x