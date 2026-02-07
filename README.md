# 🧰 GPT Toolbox

A system that augments GPT with general purpose tools. Currently implemented via gpt4 plugin that must be running on localhost.

## Recent Updates (2026)

This repository has been updated with:
- **OpenAI API v1.x+**: Migrated from deprecated v0.27.x to modern v1.x API
- **Modern Models**: Support for GPT-4o, GPT-4o-mini, GPT-4-turbo, and updated embeddings models
- **ChromaDB v0.5+**: Updated from v0.3.22 to current version with PersistentClient pattern
- **Updated Dependencies**: All dependencies updated to latest stable versions

## Setup:

1. You need the following from OpenAI:

    - **API key**: https://platform.openai.com/signup

    - **GPT4 plugin developer access**: Be subscribed to "Plus" and join the waitlist for plugin dev https://openai.com/waitlist/plugins

2. Init the project: `make init`

3. Edit the env file `.env`, put required API keys where specified

## Env:

The env 

| Name | Type | Description |
| ---- | ---- | ----------- |
| OPENAI_API_KEY | string | OpenAI API key |
| LOG_LEVEL | string | Log level (VERBOSE, NORMAL, ERROR, NONE) |
| WANDB_ENABLED | boolean | Enable Weights & Biases logging |

## Plugin

1. Start the server on **localhost:3333**:

```
make start
```

## Available Models

The toolbox now supports the following OpenAI models:

### Modern Models (Recommended)
- **GPT-4o**: Latest multimodal model with improved performance and lower cost
- **GPT-4o-mini**: Cost-effective variant of GPT-4o
- **GPT-4-turbo**: High-performance GPT-4 variant with 128K context window

### Embeddings
- **text-embedding-3-small**: Latest embedding model, cost-effective
- **text-embedding-3-large**: Highest quality embeddings
- **text-embedding-ada-002**: Legacy embedding model (still supported)

### Legacy Models
GPT-3.5-turbo and older GPT-4 variants are still supported but deprecated.

## Credits:

* OpenAI cookbook (https://github.com/openai/openai-cookbook/tree/main/examples)

* Before this was a plugin, this project started as a different way of doing "agents" (w/o the AgentExecutor) from **Langchain** (https://github.com/hwchase17/langchain)

* The executive agent is an implementation of the planning step of the system described in **HuggingGPT** (https://arxiv.org/pdf/2303.17580.pdf) and its prompt was forked from **JARVIS** (https://github.com/microsoft/JARVIS)

* Project named by gpt3.5, logo by midjourney

## Logo:

![toolbox](./src/plugin/.well-known/logo.png)
