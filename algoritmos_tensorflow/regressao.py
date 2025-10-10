import pyprind
import tensorflow as tf
from algoritmos_tensorflow.modelo      import modelo_simples
from algoritmos_tensorflow.metrics_regressao import erro_quadratico_medio

def gradiente_descedente_estocatico(x, y, model, learning_rate=0.01, epochs=1000, loss=erro_quadratico_medio, salvar_perda = False):
    pgbar = pyprind.ProgBar(epochs//2, title=model.name, monitor=True)
    perda = []
    indices = tf.range(start=0, limit=len(x), delta=1)
    for i in range(epochs):
        # Embaralhando os dados
        embaralhar = tf.random.shuffle(indices)
      

        # Atualizando pesos
        for index in embaralhar:

            xi_ = tf.reshape(x[index, :], [1, -1])
            yi_ = y[index]
            pred = model(xi_)

            # Atualizando pesos
            # Atualizando o coeficiente : w = w + lr * np.mean((y - w * x) * x)
            model.coeficiente = model.coeficiente + learning_rate * (yi_ - pred) * xi_

            # Atualizando o intercept : w = w + lr * np.mean(y - w * x) 
            model.interceptor = model.interceptor + learning_rate * (yi_ - pred)

        pgbar.update(1, item_id=f'Epoch {i} - Loss: {loss(y, pred):.5f}')
        
        if salvar_perda: perda.append(loss(y, pred))
 
    return model if not salvar_perda else (model, perda)

def gradiente(x, y, model, learning_rate=0.01, epochs=1000, loss=erro_quadratico_medio, salvar_perda = False):
    pgbar = pyprind.ProgBar(epochs, title=model.name, monitor=True)

    perda = []
    for epoch in range(epochs):
        ypred = model(x)

        # Atualizando pesos
        # Atualizando o coeficiente : w = w + lr * np.mean((y - w * x) * x)
        model.coeficiente = model.coeficiente + learning_rate * tf.reduce_mean((y - ypred) * x , axis=0)

        # Atualizando o intercept : w = w + lr * np.mean(y - w * x) 
        model.interceptor = model.interceptor + learning_rate * tf.reduce_mean(y - ypred)

        pgbar.update(1, item_id=f'Epoch {epoch} - Loss: {loss(y, ypred):.5f}')
        
        if salvar_perda:
            perda.append(loss(y, ypred))
    
    return model if not salvar_perda else (model, perda)