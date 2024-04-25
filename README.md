# NN-from-scratch
Esse repositório é fruto da série de videos do canal **sentdex**, que implementou uma rede neural em python.

Uma rede neural é um modelo que simula o modo de aprendizado humano, com suas sinapses e neurônios.

![REDE NEURAL](https://i.imgur.com/PeldHjy.png)

- Temos uma camada de entrada no início, camadas ocultas no meio (podem ser várias) e uma camada de saída no final.
- Cada bolinha azul dessas é um neurônio e eles estão conectados uns com os outros.
- Em uma rede neural o processamento do dado começa na camada de entrada, ele é passado para as camadas ocultas e depois é passado para a camada de saída.

Para treinar a rede neural, é feito o balanceamento dos **pesos** e dos **biases.**

- **Pesos:** São os padrões para a força do sinal do neurônio, esse valor vai determinar a influência dos dados de entrada no produto da saída de cada neurônio. Pense neles como os números que ajustam o impacto dos dados que estão entrando. Eles podem aumentar ou diminuir a importância de informações específicas. **Em essência, os pesos são a forma da rede neural aprender com os dados. Eles capturam as relações entre as features de entrada e de saída alvo, permitindo que a rede neural generalize e faça predições em dados que ela não viu.**

- **Bias:** Acrescentam uma camada adicional de flexibilidade para as redes neurais. Biases são constantes associadas com cada neurônio. Ao contrário dos pesos, o bias não está relacionado com dados específicos de entrada do neurônio, mas são acrescentados aos dados de saída do neurônios. Bias são como um limite, permitindo que alguns neurônios sejam ativados mesmo que a soma dos pesos dos dados que entraram não seja o suficiente por conta própria. Os biases introduzem um nível de adaptabilidade que garante que a rede possa aprender e fazer predições efetivamente. Para ficar claro, imagine um neurônio que processe o nível de brilho do pixel de uma imagem, sem o bias, esse neurônio só seria ativado quando o nível de brilho do pixel estivesse exatamente em um certo limite, mas colocando o bias, você permite que o neurônio ative mesmo quando o brilho estiver um pouco menor ou acima do limite. Essa flexibilidade é crucial porque os dados do mundo real raramente estão alinhados em um exato limite. Dessa forma, os biases permitem que os neurônios ativem em várias condições de entrada, fazendo com que as redes neurais fiquem mais robustas e capazes de lidar com dados complexos. Durante o treino, os biases são ajustados para otimizar a performance da rede neural. 

![REDE NEURAL II](https://i.imgur.com/fSBTsUP.png)

- No exemplo acima, temos 224 pesos, que são representados pelas conexões de saída → entrada entre os neurônios.
- Temos 26 biases, cada neurônio tem o seu próprio (menos os da camada de entrada).
- Nesse sentido, temos 250 parâmetros ajustáveis.

### Processo de aprendizado: Forward e Backward Propagation

#### **A. Forward Propagation**
A propagação para frente é a fase inicial de processamento dos dados de entrada através da rede neural para formar o resultado ou predição.

1. **Camada de entrada**: Os dados são inseridos na camada de entrada da rede neural.
2. **Soma ponderada dos pesos**: Cada neurônio nas camadas subsequentes calcula a soma dos pesos dos dados que recebem, em que os pesos são os parâmetros ajustáveis.
3. **Acrescentando os biases**: São acrescentados os bias associados aos neurônios para cada soma ponderada do passo anterior. Isso introduz um limite para ativação.
4. **Função de Ativação**: Os resultados da soma ponderada com o bias é passado através de uma função de ativação. Essa função determina se o neurônio deve ativar ou continuar dormente baseado no valor calculado.
5. **Propagação**: O resultado de saída uma das camadas vira a entrada para a camada seguinte, e o processo se repete até que a camada de saída produza a predição da rede.

**Essa lógica de funcionamento foi implementada no notebook do repositório.**

Abaixo segue um exemplo visual de funcionamento de rede neural:

![REDE NEURAL GATO](https://i.imgur.com/2kKW6Q5.png)
![REDE NEURAL GATO](https://i.imgur.com/AZvTtgI.png)
![REDE NEURAL GATO](https://i.imgur.com/iK7k2IV.png)

A foto de um gato foi inserida na rede neural e ela identificou ele corretamente como um gato, pode-se dizer que o neurônio de gato na camada de saída foi o mais forte.

#### **B. Backward Propagation**

Depois que a rede neural fez uma predição, precisamos avaliar o quão certeira ela foi e também precisamos fazer ajustes para melhorar futuras predições.

1. **Cálculos dos erros**: A predição da rede neural é comparado com o fato real, o erro resultante, chamado de **loss** ou **cost**, mede a disparidade entre predição e realidade.
2. **Gradiente Descendente**: A propagação para trás envolve minimizar esse erro. Para fazer isso, a rede calcula o gradiente do erro considerante os pesos e as biases. Esse gradiente aponta na direção da diminuição mais acentuada do erro.
3. **Atualização de Peso e Bias**: A rede neural usa as informações que obteve com o gradiente descendente para ajustar os pesos e os biases de toda a rede neural. O objetivo é encontrar os valores que minimizam o erro.
4. **Processo de Iteração**: Esse processo de forward e backward propagation é repetido iterativamente em lotes de dados de treino, com cada iteração, os pesos e os biases da rede se aproximam de valores que minimizam o erro.

**Em resumo, a propagação para trás ajusta os parâmetros da rede, fazendo com que fiquem com mais acurácia. Esse processo de iteração continua até que a rede alcance um nível satisfatório de performance nos dados de treino**

### **Cálculo da Saída do Neurônio de forma visual**
 A saída do neurônio é dada por:  

\[
\text{Saída Neurônio} = \text{Input} \times \text{Peso} + \text{Bias}
\]  
Ela se assemelha a uma função de primeiro grau, esse seria o comportamento dela:

![NN_formula](https://i.imgur.com/lzEKc4S.gif)

### **Shape**:
Para trabalhar com redes neurais e deep learning, precisamos entender o conceito de **shape** ou formato nas linguagens de programação.

No contexto de **deep learning em python**, vamos ver as formas que o **numpy** reconhece:

![shape](https://i.imgur.com/sNTvXjn.png)

Acima temos uma lista com 4 elementos, de uma dimensão, então é um vetor. Ela tem 4 elementos, então o shape dela é (4, ).

![shape](https://i.imgur.com/rjvZZn2.png)

Acima temos uma lista com duas listas dentro, cada uma com 4 elementos. Isso é uma matriz, o shape dela é (2, 4).

![shape](https://i.imgur.com/5qYQjEk.png)

Acima temos uma lista, dentro dessa lista temos 3 outras listas, dentro dessas 3 outras listas, temos 2 listas em cada, em cada uma dessas duas listas temos 4 elementos.

Temos outra forma que é o **tensor**, de forma bem rasa, um tensor é um objeto que pode ser representado como um array, ele não é um simples array, mas no contexto de programação e deep learning ele pode ser representado e trabalhado como um array.  

### **Dot product**

Pegando o conceito dos tipos de array acima, podemos fazer operações entre eles, no caso do **Dot Product**, estamos falando da **multiplicação** de dois arrays, e ele nos retorna somente um valor. Vale ressaltar que o dot product vem da biblioteca do numpy.

![dot_product](https://i.imgur.com/xh5txKP.png)

### Batches

Usamos lotes nos inputs na camada de entrada da rede porque assim facilita o aprendizado para o modelo. Se usarmos apenas uma entrada por vez para treinar o modelo, ele terá mais dificuldades para fazer os ajustes. Se usarmos mais entradas por vezes, o modelo irá se ajustar com mais facilidade.

![dot_product](https://i.imgur.com/dkR4r0K.gif)

No exemplo acima, os pontos que surgem são os dados que pedimos para a rede se ajustar, como estamos passando dados de entrada com lotes de tamanho 1, percebe-se que a linha se movimenta e muito.

![dot_product](https://i.imgur.com/JESirG6.gif)

Já nesse outro exemplo, a linha vai se adaptando, e quando se ajusta totalmente, percebe-se que ela vai se movendo muito pouco, isso porque estamos agora passando dados de entrada com lotes com tamanho 32.

### Funções de Ativação

Como vimos acima, os resultados de uma rede neural possuem uma relação linear, para introduzirmos uma **Não-Linearidade** na rede neural e assim possibilitarmos que ela compreenda melhor relações mais complexas nos dados, usamos as **funções de ativação**.

A ideia é que as funções de ativação são aplicadas nos resultados dos neurônios, após todas as operações.

![products](https://miro.medium.com/v2/resize:fit:828/format:webp/1*bUNxrEy2KjKNrwMRo8eS9w.png)

Depois de pegar os resultados $z_{1}$ e $z_{2}$ aplicamos eles na função de ativação 𝜎.

![activation function >](https://miro.medium.com/v2/resize:fit:278/format:webp/1*9DfvAg_pENO5MX0ELeY_kg.png)


