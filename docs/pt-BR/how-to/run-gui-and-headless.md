---
title: Executar nos Modos GUI e Headless
---

# Executar nos Modos GUI e Headless

O Koharu pode ser executado como um app desktop normal ou como um servidor local headless com Web UI. Os dois modos usam o mesmo runtime local e o mesmo servidor HTTP.

## O que continua igual entre os modos

Não importa como você inicia o Koharu, o modelo de runtime é o mesmo:

- o servidor faz bind em `127.0.0.1` por padrão (sobrescreva com `--host`)
- a UI e a API são servidas pelo mesmo processo local
- o pipeline de páginas, o carregamento de modelos e as exportações usam os mesmos caminhos de código internos

Por isso a edição no desktop e a automação headless ficam sempre alinhadas.

## Resumo dos modos

| Modo | Janela desktop | Servidor local | Uso típico |
| --- | --- | --- | --- |
| Desktop | sim | sim | edição interativa normal |
| Headless | não | sim | Web UI local, scripting, automação |

## Executar o app desktop

Abra o Koharu normalmente a partir do atalho do aplicativo instalado.

Mesmo no modo desktop, o Koharu ainda sobe um servidor HTTP local internamente. A janela embutida conversa com esse servidor local em vez de chamar o pipeline diretamente.

Este é o modo padrão e a melhor escolha para a maioria dos usuários.

## Executar em modo headless

O modo headless sobe o servidor local sem abrir a GUI desktop.

```bash
# macOS / Linux
koharu --port 4000 --headless

# Windows
koharu.exe --port 4000 --headless
```

Depois da inicialização, abra a Web UI em `http://localhost:4000`.

O modo headless fica em primeiro plano até você parar, normalmente com `Ctrl+C`.

## Executar com uma porta fixa

Por padrão, o Koharu usa uma porta local aleatória. Use `--port` quando precisar de um endereço estável para bookmarks, scripts, reverse proxies ou clientes da API.

```bash
# macOS / Linux
koharu --port 9999

# Windows
koharu.exe --port 9999
```

Se você não especificar `--port`, o Koharu ainda inicia o servidor, mas a porta escolhida é dinâmica.

## Vincular a um endereço fora do loopback

Por padrão, o servidor faz bind em `127.0.0.1`, o que significa que apenas a mesma máquina consegue alcançá-lo. Passe `--host` para fazer bind em outro lugar.

```bash
koharu --host 0.0.0.0 --port 4000 --headless
```

Isso é útil em containers, VMs ou setups de desenvolvimento remoto, onde o cliente desktop vive em um host diferente do processo do Koharu. Qualquer coisa diferente de `127.0.0.1` é uma escolha deliberada — não há autenticação embutida na API local, então só defina `--host` quando você realmente quiser acesso fora do loopback e tiver seus próprios controles de acesso no lugar.

## Conectar à API local

Quando o Koharu está rodando em uma porta fixa, os endpoints principais são:

- Web UI: `http://localhost:9999/`
- RPC / HTTP API: `http://localhost:9999/api/v1`

Substitua `9999` pela porta que você escolheu.

Como o Koharu faz bind em loopback, esses endpoints são locais por padrão. Se quiser acesso a partir de outra máquina, você precisa expor essa porta por conta própria via sua configuração de rede.

Para detalhes por endpoint, veja a [Referência da HTTP API](../reference/http-api.md).

## Forçar o modo CPU

Use `--cpu` quando quiser desabilitar inferência por GPU explicitamente.

```bash
# macOS / Linux
koharu --cpu

# Windows
koharu.exe --cpu
```

Isso é útil para testes de compatibilidade, problemas de driver ou debug de baixo risco quando o setup de GPU está incerto.

## Apenas baixar dependências de runtime

Use `--download` se quiser que o Koharu pré-baixe as dependências de runtime e encerre sem iniciar o app.

```bash
# macOS / Linux
koharu --download

# Windows
koharu.exe --download
```

Na implementação atual, esse caminho inicializa:

- bibliotecas de runtime usadas pela stack de inferência local
- os modelos padrão de visão e OCR

Ele não pré-baixa todas as LLMs locais opcionais de tradução. Essas continuam sendo baixadas quando você as seleciona em Settings.

## Habilitar saída de debug

Use `--debug` quando quiser inicialização orientada a console com output de logs.

```bash
# macOS / Linux
koharu --debug

# Windows
koharu.exe --debug
```

No Windows, execuções de debug e headless também influenciam como o Koharu anexa ou cria uma janela de console.

## Armazenamento de credenciais

Por padrão, o Koharu armazena API keys fora de `config.toml`. macOS e Windows usam o keyring do sistema. Linux usa o armazenamento local de credenciais do Koharu no diretório de dados do app com permissões somente para o usuário dono; esse armazenamento no Linux depende das permissões do filesystem em vez de criptografia em nível de sistema operacional.

Execuções headless e em container usam o mesmo comportamento de armazenamento de credenciais do app desktop. Em containers, mantenha o diretório de dados do app em um volume persistente se quiser que as API keys salvas sobrevivam à substituição do container.
