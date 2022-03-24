sudo apt-get install -y python3-pip python3-tk
printf "\nalias python='python3'" >> ~/.bashrc

sudo pip3 install virtualenv virtualenvwrapper
source ~/.bashrc

printf "\nexport WORKON_HOME=$HOME/.virtualenvs\nexport VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3\nsource /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc
source ~/.bashrc
mkvirtualenv cvML
workon cvML

pip3 install -r requirements.txt 
