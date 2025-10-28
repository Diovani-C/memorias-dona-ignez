# Projeto Memórias da Vó Ignez

Um website estático dedicado a preservar e celebrar as memórias da minha falecida avó.

## Sobre o Projet

Este projeto é uma homenagem pessoal e um esforço para criar um arquivo digital permanente da vida da minha avó. O objetivo era reunir fotografias de diversas fontes, restaurá-las e apresentá-las em um website simples, rápido e acessível para toda a família.

O site é totalmente estático, construído com [AstroJS](https://astro.build/), o que garante excelente performance e facilidade de manutenção. A hospedagem é feita em um bucket do [Google Cloud Storage](https://cloud.google.com/storage), configurado para servir conteúdo estático.

## Tecnologias Principais

- **Frontend:** AstroJS
- **Hospedagem:** Google Cloud Storage (GCS)
- **Processamento de Imagem (Bibliotecas):** [spandrel](https://pypi.org/project/spandrel/) (Python), [face-api.js](https://justadudewhohacks.github.io/face-api.js/docs/index.html) (JavaScript)
- **Processamento de Imagem (Ferramentas):** Darktable, Docker
- **Inteligência Artificial:** API do Gemini (para restauração e geração de metadados)

## O Processo de Construção

O maior desafio do projeto não foi o desenvolvimento do site em si, mas o complexo pipeline de coleta, restauração e catalogação das imagens.

### 1. Coleta e Digitalização

As imagens foram reunidas de múltiplas fontes:

- **Arquivos Físicos:** Uma grande parte das fotos estava na casa da minha avó. Elas foram cuidadosamente digitalizadas usando o scanner de uma impressora.
- **Arquivos Digitais:** Outras imagens foram contribuições de parentes, vindas de seus arquivos digitais pessoais.

### 2. Restauração e Pós-Processamento

Cada imagem passou por uma avaliação para determinar o tratamento necessário:

- **Imagens em Bom Estado:** Para fotos sem danos significativos, o tratamento foi mínimo. Elas foram mantidas em seu estado original ou passaram apenas por uma leve correção de cor e contraste utilizando o software [Darktable](https://www.darktable.org/).

- **Restauração Automatizada (Spandrel):** Para imagens danificadas (com ruído, desfoque, baixa resolução ou em P&B), utilizei a biblioteca [spandrel](https://pypi.org/project/spandrel/) do Python. Criei um ambiente isolado com Docker para executar os scripts de restauração, testando vários modelos pré-treinados para:
  - Remoção de ruído (denoising)
  - Aumento de resolução (upscaling)
  - Remoção de desfoque (deblurring)
  - Colorização.

- **Restauração Avançada (IA Generativa):** Nos casos em que os modelos do `spandrel` não produziam um resultado satisfatório, criei um segundo script. Este script utilizava um prompt customizado e a API do "gemini nano banana" para tentar uma restauração mais complexa da imagem.

### 3. Geração de Metadados e Descrições

Para enriquecer o acervo, era crucial adicionar contexto e identificar as pessoas nas fotos.

- **Descrições de Imagens (Alt Text):** Para garantir a acessibilidade e contextualizar cada foto, utilizei a API do Gemini para gerar descrições detalhadas (texto alternativo) para cada imagem.

- **Identificação Facial (Tagging):** Para identificar as pessoas nas fotos, criei uma ferramenta interna (uma página não listada no site, acessível em `/face`).
  - **Tecnologia:** A página utiliza a biblioteca [face-api.js](https://justadudewhohacks.github.io/face-api.js/docs/index.html).
  - **Processo:** Eu fazia o upload de um lote de imagens. A `face-api.js` processava tudo localmente no navegador, detectando os rostos. Eu então inseria os nomes para cada rosto detectado.
  - **Resultado:** Ao final do processo, a ferramenta permitia o download de um arquivo `json`. Esse JSON foi posteriormente utilizado para gerar os metadados das imagens no site, permitindo saber quem está em cada foto.

## Aprendizados e Conclusão

Este foi um projeto profundamente pessoal e, ao mesmo tempo, um grande desafio técnico. Aprendi muito sobre processamento digital de imagens, desde correções de cores básicas até a aplicação de modelos complexos de IA para restauração e análise.

Foi uma experiência incrivelmente gratificante poder usar minhas habilidades técnicas para criar uma homenagem tão significativa e duradoura para a minha avó, preservando sua história para as futuras gerações da nossa família.
