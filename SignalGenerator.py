import random as rnd
import logging
import json
import numpy as np

logging.basicConfig(
    format='%(levelname)s [%(module)s]: %(message)s'
)
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

def generate_random_signal(min_val, max_val, nof_samples):
    choices = list(range(min_val, max_val+1))
    weights = list(0.05/abs(i) if i !=0 else 1.0 for i in choices)
    return list(baker_algorithm(choices, weights) for i in range(nof_samples))

def save_signal(signal, output_path):
    try:
        with open(output_path, 'w') as f:
            f.write(json.dumps(signal, indent=2))
    except Exception as e:
        _logger.error(f'Could not save the signal in file {output_path}\n{e}')

def read_signal(input_path):
    try:
        with open(input_path) as f:
            signal = json.load(f)
        return signal
    except Exception as e:
        _logger.error(
            f'Could not import the signal from file {input_path}\n{e}')
        return None

def sum_list(l):
    result = 0    
    for i in l: 
        result += i
    return result

def normalize(p_k, M, P):
    return M*(p_k/sum_list(P))

def normalize_all(P, M):
    P_norm = []
    for i in P:
        P_norm.append(normalize(i, M, P)) 
    return P_norm   

def check_input(A, P):
    if(len(A) != len(P)):
        logging.error("Inconsistent length for the input arrays!")        
        exit()
    else:
        for p in P:
            if(not(isinstance(p, float))):
                logging.error("Probability array must contain only float type values!")   
                logging.info("Found: " + str(p))                
                exit()
            if(p < 0):
                logging.error("Weights must be positive float numbers!")
                logging.info("Found: " + str(p))
                exit()
            if(p > 1):
                logging.warning("Weights are recommended to be less than or equal with 1!")
                logging.info("Found: " + str(p))

def baker_algorithm(A, P):
    check_input(A, P) 
    N = len(A)  
    M = int(sum_list(P)*10)
    P = normalize_all(P, M)
    u = rnd.uniform(0, 1) 
    v = 0
    U = []    
    S = [0]*N
    for n in range(N):
        v = v + P[n]
        while(u < v):
            U.append(A[n])
            S[n] += 1
            u = u + 1
    return rnd.choice(U)


def generate_step_signal(size, freq, max_val=1, min_val=0):
    new = []    
    for i in range(size):
        if i%freq < freq/2:
            new.append(min_val)
        else:
            new.append(max_val)

    return np.array(new)


