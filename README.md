# Tarea 1
## Redes Neuronales y Programación Genética

Se hizo:

## Implementación de una Red Neuronal. 

Esta cuenta con layers (hidden o output) y estos layers con cierta cantidad de neuronas.

## Uso de la red neuronal para la clasificación de dígitos escritos a mano. 

El dataset usado venía con los datos preprocesados, por lo cual estos consisten en un input de 64 números enteros entre 0 y 16.
Estos números representan un dígto escrito a mano, por lo cual los posibles outputs corresponden a números entre 0 y 9, 
por lo que se tendrían 10 posibles clases.

Se prueba con varias "arquitecturas" de red, con las cuales se obtienen los siguientes resultados de precisión:

- 1 hidden layer con 20 neurones: 0.919
- 1 hidden layer con 30 neurones: 0.926
- 1 hidden layer con 35 neurones: 0.927
- 2 hidden layers con 40 y 20 neurones: 0.922
- 2 hidden layers con 35 y 35 neurones: 0.931 (*)
- 2 hidden layers con 35 y 30 neurones: 0.910
- 2 hidden layers con 35 y 30 neurones: 0.923

La configuración con mejor performance fue la de 2 hidden layers.


