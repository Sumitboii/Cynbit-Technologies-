.PHONY: install setup data train api ui test eval lint pdf zip

# Install core dependencies
install:
	pip install -r requirements.txt

# Install optional dependencies (MiniLM + OpenAI)
install-optional:
	pip install -r requirements-optional.txt

# Full setup for a fresh clone: install deps + generate data
setup: install data
	@echo "Setup complete. Run 'make eval' to verify."

# Generate synthetic data (must run before tests or eval)
data:
	python data/generate_reviews.py

# Train TF-IDF+LR and MiniLM+LR models, produce model comparison table
train:
	python -m src.models.train_and_eval

# Train TF-IDF+LR only (faster, no sentence-transformers needed)
train-fast:
	python -m src.models.train_and_eval --no-miniLM

# Run evaluation on synthetic test set
eval:
	python -m src.evaluate

# Run evaluation on manually-authored gold set
eval-gold:
	python -m src.evaluate_handwritten

# Start FastAPI server
api:
	uvicorn app.main:app --reload --port 8000

# Start Streamlit dashboard
ui:
	streamlit run app/dashboard.py

# Run tests (requires data to be generated first: make data)
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Lint
lint:
	ruff check src/ tests/ app/

# Export markdown reports to PDF
pdf:
	python export_pdfs.py

# Regenerate the submission zip (excludes .git, __pycache__, .kiro)
zip:
	python -c "\
import zipfile, pathlib; \
src = pathlib.Path('.'); \
dest = pathlib.Path('../cynbit_ai_revio_v2.zip'); \
SKIP = {'.git','__pycache__','.pytest_cache','.kiro','models'}; \
count = 0; \
zf = zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED); \
[zf.write(f, pathlib.Path('cynbit.ai') / f.relative_to(src)) or True \
  for f in src.rglob('*') \
  if f.is_file() \
  and not set(f.relative_to(src).parts) & SKIP \
  and f.suffix not in {'.pyc','.pyo','.pkl'}]; \
zf.close(); \
print('Zip created:', dest)"
