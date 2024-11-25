# Steam dataset analytics

Para executar o projeto, tenha o [python](https://www.python.org/downloads/) e [pip](https://pip.pypa.io/en/stable/cli/pip_install/) instalado, e siga os seguintes passos:

1. Clone o repositório:
```bash
git clone https://github.com/metehus/steam-dataset-analytics
cd steam-dataset-analytics
```
2. Crie um penv para o projeto:
```bash
py -m venv venv
```
3. Entre no ambiente do python. No windows, via PowerShell, use:
```bash
.\venv\Scripts\Activate.ps1
```
  ps: Se der erro de permissão de execução, rode o seguinte comando em um terminal powershell como administrador: ` Set-ExecutionPolicy -ExecutionPolicy RemoteSigned`

4. Dentro do penv, instale as dependencias:
```bash
pip install -r requirements.txt
```
5. Inicie a aplicação:
```bash
py manage.py runserver
```
6. O terminal irá dizer o endereço local para acessar, mas geralmente vai ser http://localhost:8000/
7. Para enviar e treinar o dataset, baixe-o do kaggle: https://www.kaggle.com/datasets/trolukovich/steam-games-complete-dataset