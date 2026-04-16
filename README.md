# LLM Inference Benchmark: vLLM vs Hugging Face

Este repositório contém uma suíte completa de benchmark (testes de estresse) projetada para avaliar e comparar empiricamente o desempenho de inferência da biblioteca de altíssima performance **vLLM** contra a implementação padrão (vanilla) do **Hugging Face Transformers**.

O objetivo desta suíte é fornecer dados que atestam a diferença brusca de processamento proporcionada pelo gerenciamento otimizado de cache (PagedAttention) e paralelismo de alto nível, exibindo métricas de RPS (Requisições Por Segundo), TPS (Tokens Por Segundo) e a taxa de Ocupação da GPU.

## 🛠️ O que tem aqui dentro?

- `install_vllm.sh` & `requirements.txt`: Instala todas as dependências do ambiente.
- `start_server.sh`: Inicia o motor nativo VLLM hospedando a API OpenAI na porta 8000.
- `hf_api_server.py` & `start_hf_server.sh`: Sobem um servidor mock usando FastAPI e o pipeline Hugging Face para simular o mesmo caminho da API OpenAI (para uma comparação 1:1 rigorosa e justa).
- `benchmark.py`: A estrela do projeto. Orquestrador "multi-threading" que bombardeia o servidor e ao mesmo tempo mede a utilização de sua GPU rodando o `nvidia-smi` em segundo plano.

---

## 🚀 Como Executar

### 1. Inicializando o Ambiente
O único passo de configuração necessário é executar o instalador (recomendamos o ambiente Linux / Container da Vast.ai):
```bash
bash install_vllm.sh
```

### 2. Iniciar o Motor Alvo (Background)
Em uma janela de terminal, inicie o servidor da sua escolha:

Para testar as otimizações revolucionárias do **vLLM**:
```bash
bash start_server.sh
```

Ou, caso queira executar no **Hugging Face** para obter sua baseline:
```bash
bash start_hf_server.sh
```
*(Nota: O servidor ficará exposto em `http://localhost:8000`)*

### 3. Disparar a Carga do Benchmark

Tendo habilitado o servidor acima, abra um novo shell/terminal e execute o teste. Você pode (e deve) usar o parâmetro `--engine` para categorizar o nome dos seus resultados de saída.

Para rodar com o motor VLLM:
```bash
python benchmark.py --engine VLLM
```

Para rodar contra a implementação original livre do HF:
```bash
python benchmark.py --engine HuggingFace-Original
```

---

## 📊 Relatórios e Saídas

Os resultados serão mostrados imediatamente e de maneira estética no seu terminal com tabelas construídas nativamente pelo `rich` do Python. O Output mostrará o impacto que o aumento de **Concorrência (`requests`)** e de **Profundidade de Saída (`max_tokens`)** gera em RPS, TPS e Custo (% de ocupação da GPU hardware).

Ao final de cada execução, a suíte irá gerar atomaticamente e salvar um elegante artefato visual em Markdown na raiz:
`benchmark_results_VLLM.md`

Use este projeto em instâncias e mostre ao mundo as suas evidências métricas!
