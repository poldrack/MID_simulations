all: create install

create:
	micromamba create -y -f env.yml

install:
	micromamba run -n midsims pip install -r requirements.txt


test:
	coverage run -m pytest --disable-warnings
	coverage report -m

format-notebook: test
	git add MID_sim_group_fullmodel_manuscript.ipynb
	git commit -m"Formatting notebook using blue"
	jupytext --to py:percent MID_sim_group_fullmodel_manuscript.ipynb
	blue MID_sim_group_fullmodel_manuscript.py
	jupytext --to ipynb MID_sim_group_fullmodel_manuscript.py
	git add MID_sim_group_fullmodel_manuscript.ipynb
	git commit -m"Formatted notebook using blue"
