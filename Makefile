install:
	pip install --upgrade pip
	pip install -r requirements.txt

format:
	black *.py

train:
	python train.py

eval:
	echo ## Model Metrics > report.md
	type Results\metrics.txt >> report.md
	echo. >> report.md
	echo ## Confusion Matrix Plot >> report.md
	echo ![Confusion Matrix](./Results/model_results.png) >> report.md
	cml comment create report.md

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git add Results/metrics.txt Results/model_results.png Model/breast_cancer_pipeline.skops
	git commit -am "Update with new results"
	git push --force origin HEAD:update

prepare-app:
	if not exist App\Model mkdir App\Model
	if not exist App\Data mkdir App\Data
	copy Model\breast_cancer_pipeline.skops App\Model\
	copy Data\data.csv App\Data\

hf-login:
	pip install -U "huggingface_hub==0.24.0"
	python -c "from huggingface_hub import login; login('$(HF)')"

push-hub: prepare-app
	python -c "from huggingface_hub import HfApi; api = HfApi(); api.upload_folder(folder_path='./App', repo_id='karima15/Breast-Cancer-App', repo_type='space', commit_message='Deploy model')"

deploy: hf-login push-hub

.PHONY: install format train eval update-branch prepare-app hf-login push-hub deploy