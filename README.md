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