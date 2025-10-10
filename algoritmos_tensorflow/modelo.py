import tensorflow as tf

class modelo_simples(tf.Module):
    def __init__(self, name='modelo_1', features=1, seed=32):
        super().__init__(name)
        # Seed 
        tf.random.set_seed(seed)

        self.interceptor = tf.Variable([0.0], name='interceptor', dtype=tf.float32, trainable=True)
        self.coeficiente = tf.Variable(tf.random.uniform(shape=[1, features]), name='coeficiente', dtype=tf.float32, trainable=True)
    
    def __call__(self, X):
        return tf.matmul(X, self.coeficiente, transpose_b=True) + self.interceptor
    
    def summary(self):
        return f'Intercept: {self.interceptor.numpy().tolist()} | Coeficiente: {self.coeficiente.numpy().tolist()}'
