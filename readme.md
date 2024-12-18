# Aplicação Bancária
 
Aplicação desenvolvida no âmbito do MESI, disciplina de Tecnologias Biométricas, com implementação de reconhecimento facial e keystroke dynamics.
 
## Instalação dos requirements
 
Instalação dos requirements na máquina host.
```bash
sudo apt update
pip install -r requirements.txt
```
 
## Docker
 
Para instalar o docker package fazer:
```bash
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable docker --now
sudo usermod -aG docker $USER
````
 
Para importar o docker fazer:
```bash
tar --no-same-owner -xjvf meudocker.tar.bz2
docker import meudocker.tar tbmesi #sha256:388ca49fa3fac1c5a33ed5fa8ab21fc9811cab8fdb1426dfa13b085c6bb66220
docker run --name dockerfinal -it tbmesi /bin/bash 
```
