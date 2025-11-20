# Makefile pour ML-CICD-BreastCancer

install:
	pip install --upgrade pip && \
	pip install -r requirements.txt

format:
	black *.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md
	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./Results/confusion_matrix.png)' >> report.md
	cml comment create report.md

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new results"
	git push --force origin HEAD:update

# --- Deployment commands (corrigés pour Windows et Linux) ---

prepare-app:
	mkdir -p App || true
	cp Model/breast_cancer_model.skops App/ || copy Model\breast_cancer_model.skops App\ || true
	cp Data/data.csv App/ || copy Data\data.csv App\ || true

hf-login:
	pip install -U huggingface_hub
	python -c "from huggingface_hub import login; login('$(HF)')"

push-hub: prepare-app
	python -c "from huggingface_hub import HfApi; api = HfApi(); api.upload_folder(folder_path='./App', repo_id='karima15/Breast-Cancer-App', repo_type='space', commit_message='Sync App files')"

deploy: hf-login push-hub

.PHONY: install format train eval update-branch prepare-app hf-login push-hub deploy