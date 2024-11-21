# Steam dataset analytics

Para executar o projeto, tenha o python e pip instalado, e siga os seguintes passos:

1. Crie um penv para o projeto:
```bash
py -m venv venv
```
2. Entre no ambiente do python. No windows, via PowerShell, use:
```bash
.\venv\Scripts\Activate.ps1
```
  ps: Se der erro de execução, rose o seguinte comando em um terminal powershell como administrador: ` Set-ExecutionPolicy -ExecutionPolicy RemoteSigned`

3. Dentro do penv, instale as dependencias:
```bash
pip install -r requirements.txt
```
4. Inicie a aplicação:
```bash
py manage.py runserver
```