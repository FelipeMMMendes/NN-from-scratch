# NN-from-scratch
Esse repositório é fruto da série de videos do canal **sentdex**, que implementou uma rede neural em python.

Uma rede neural é um modelo que simula o modo de aprendizado humano, com suas sinapses e neurônios.

![REDE NEURAL](https://i.imgur.com/PeldHjy.png)

- Temos uma camada de entrada no início, camadas ocultas no meio (podem ser várias) e uma camada de saída no final.
- Cada bolinha azul dessas é um neurônio e eles estão conectados uns com os outros.
- Em uma rede neural o processamento do dado começa na camada de entrada, ele é passado para as camadas ocultas e depois é passado para a camada de saída.

Abaixo segue um exemplo de funcionamento de rede neural:

![REDE NEURAL GATO](https://i.imgur.com/2kKW6Q5.png)
![REDE NEURAL GATO](https://i.imgur.com/AZvTtgI.png)
![REDE NEURAL GATO](https://i.imgur.com/iK7k2IV.png)

A foto de um gato foi inserida na rede neural e ela identificou ele corretamente como um gato, pode-se dizer que o neurônio de gato na camada de saída foi o mais forte.

Para treinar a rede neural, é feito o balanceamento dos **pesos** e dos **biases.**

- **Pesos:** São os padrões para a força do sinal do neurônio, esse valor vai determinar a influência dos dados de entrada no produto da saída de cada neurônio. Pense neles como os números que ajustam o impacto dos dados que estão entrando. Eles podem aumentar ou diminuir a importância de informações específicas. **Em essência, os pesos são a forma da rede neural aprender com os dados. Eles capturam as relações entre as features de entrada e de saída alvo, permitindo que a rede neural generalize e faça predições em dados que ela não viu.**
- **Bias:** Acrescentam uma camada adicional de flexibilidade para as redes neurais. Biases são constantes associadas com cada neurônio. Ao contrário dos pesos, o bias não está relacionado com dados específicos de entrada do neurônio, mas são acrescentados aos dados de saída do neurônios. Bias são como um limite, permitindo que alguns neurônios sejam ativados mesmo que a soma dos pesos dos dados que entraram não seja o suficiente por conta própria. Os biases introduzem um nível de adaptabilidade que garante que a rede possa aprender e fazer predições efetivamente. Para ficar claro, imagine um neurônio que processe o nível de brilho do pixel de uma imagem, sem o bias, esse neurônio só seria ativado quando o nível de brilho do pixel estivesse exatamente em um certo limite, mas colocando o bias, você permite que o neurônio ative mesmo quando o brilho estiver um pouco menor ou acima do limite. Essa flexibilidade é crucial porque os dados do mundo real raramente estão alinhados em um exato limite. Dessa forma, os biases permitem que os neurônios ativem em várias condições de entrada, fazendo com que as redes neurais fiquem mais robustas e capazes de lidar com dados complexos. Durante o treino, os biases são ajustados para otimizar a performance da rede neural.

![REDE NEURAL II](https://i.imgur.com/fSBTsUP.png)

- No exemplo acima, temos 224 pesos, que são representados pelas conexões de saída → entrada entre os neurônios.
- Temos 26 biases, cada neurônio tem o seu próprio (menos os da camada de entrada).
- Nesse sentido, temos 250 parâmetros ajustáveis.