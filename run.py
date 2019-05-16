# -*- coding: utf-8 -*-
"""

O módulo runpy é usado qdo vc chama "python -m modulo".
To usando uma função dele pra chamar do jeito que chama na linha de comando.

https://docs.python.org/2/library/runpy.html

Copyright (C) 2019 Cassio Fraga Dantas

SPDX-License-Identifier: AGPL-3.0-or-later

"""

import runpy
import sys

# Coloque nessa lista seus argumentos da commandline, dentro de aspas e 
# separados por virgula (ao invez de espaços)
sys.argv += ['15']#['11','-algo','FISTA','-decay','0.1','-extra','scr_type','GAP']
# seleciona um console python normal e da ctrl-f5 ( ou clica no playpause azul)
# nao esquece de clicar no stop antes de lançar de novo

runpy._run_module_as_main("experiments")
