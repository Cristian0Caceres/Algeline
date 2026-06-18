import numpy as np

def Pagerank(_Matrix, _I = 10, _P = 0.85):
    _len = len(_Matrix)
    PR = np.ones(_len) / _len

    for iteracion in range(_I):

        PR = _P * _Matrix @ PR + (1-_P)/_len

        print(
            f"Iteración {iteracion+1}:",
            PR
        )
    return PR