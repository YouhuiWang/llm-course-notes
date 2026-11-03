.PHONY: help serve build figures pdf check clean all

PY := .venv/bin/python
MKD := .venv/bin/mkdocs

help:
	@echo "make serve     启动本地预览 http://127.0.0.1:8000"
	@echo "make build     构建静态站点到 site/"
	@echo "make figures   重新生成全部插图（scripts/figs/*.py）"
	@echo "make pdf       生成 dist/大模型发展与相关技术详解.pdf"
	@echo "make check     检查死链、术语一致性、公式编号"
	@echo "make all       figures + build + pdf"
	@echo "make clean     清理构建产物"

serve:
	$(MKD) serve --dev-addr 127.0.0.1:8000

build:
	$(MKD) build --strict

figures:
	@for f in scripts/figs/*.py; do \
		case "$$f" in *style.py|*_util.py) continue;; esac; \
		echo "▸ $$f"; $(PY) "$$f" || exit 1; \
	done
	@echo "✓ 插图已更新到 figures/"

pdf: build
	@bash scripts/build_pdf.sh

pipeline:
	$(PY) scripts/pipeline.py $(L) $(ARGS)

lab1:
	$(PY) labs/lab1_attention/main.py

check:
	$(PY) scripts/check_links.py
	$(PY) scripts/check_terms.py
	$(PY) scripts/audit.py

all: figures build pdf

clean:
	rm -rf site dist .cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
