# 1. Introduction

lo stato è un vettore di features (le distanze dal muro nel caso della macchina). 

L'input è il vettore di features, l'output è il q value di varie azione 

Il learning è effettuato considerando lo squared bellman error

        ((R(s,a,s')+ gamma * max_a' Q(s',a')) - Q(s,a))^2

la policy action sarà l'azione con il q value più alto


1. prendi qualche azione a_i e osserva (s_i, a_i, s'_i, r_i)
2. y_i = r_i + gamma max_a' Q_phi (s'_i, a') # dove il Q_phi è output rete 
3. phi <- phi - alpha * d (W_phi * s_i + b_phi - y_i)^2 / d phi

# 2. Deep Q Learning

La funzione approssimata dalla rete è il q value per ogni azione, dato in input un certo stato.

Solitamente, il più alto q value è associato all'azione migliore da fare. 

L'azione viene inviata al simulator e da questo si ottiene una reward, e contemporaneamente si ottiene lo stato successivo che viene dato in input alla rete.

Algoritmo:

- inizializza replay memory buffer D a size N
- inizializza action value function Q con pesi random
- inizializza la target value function Q con pesi $\theta= 0$
- for episode = 1,..,M do:
  - inizializza lo stato iniziale $s_1$
  - for t = 1,..,T:
    - con probabilità $\epsilon$, scegli azione random $a_t$
    - altrimenti $a_t= \argmax_a Q(s_t, a)$
    - esegui l'azione a nel simulatore e ottieni la ricompensa $r_t$ e lo stato successivo $s_{t+1}$
    - memorizza nel replay buffer (D) la tupla ($\phi_t,a_t,r_t,\phi_{t+1}$) (dove $\phi_t$ è lo stato $s_t$); solitamente si aggiunge un elemento finale alla tupla per indicare se è uno stato finale o meno)
    - sample un minibatch di transizioni ($\phi_t,a_t,r_t,\phi_{t+1}$) da D
    - si definisce il target 
    - si effettua la gradient descent su Q rispetto a $\theta$ (non effettueremo lo square)
    - ogni C step sostituiamo $\hat{Q}=Q$, ovvero, usiamo i parametri ottenuti durante la discesa del gradiente


# 3. ATARI

Nel caso del gioco breakout, un'osservazione consiste in m frame successvi (resize a 84x84), convertiti in scala di grigi. Inoltre, le reward sono posti a -1 e 1.

Poiché si "vedono" k frame alla volta, l'azione viene presa per questi k frame e applicati per ognuno di essi.

Inoltre, all'inizio di ogni episodio si possono effettuare al massimo 30 azioni nulle, aggiungendo stocasticità.

k = 4

<hr>

### 4. Install package (for colab)
setup.py file in the package directory

    from setuptools import setup, find_packages

    setup(
        name='my_package',
        version='0.1',
        packages=find_packages(),
    )

run this command

    python setup.py bdist_wheel
  
A directory named dist will be created. Use the .whl file in the dist directory to install the package.

    pip install dist/my_package-0.1-py3-none-any.whl
