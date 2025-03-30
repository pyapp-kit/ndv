.PHONY: docs docs-serve

# Define a comma-separated list of extras.
comma := ,
extras ?= pyqt,pygfx
EXTRA_FLAGS := $(foreach extra, $(subst $(comma), ,$(extras)),--extra=$(extra))
GROUP_FLAGS := $(foreach extra, $(subst $(comma), ,$(groups)),--group=$(extra))

ifeq ($(filter pyqt pyside, $(subst $(comma), ,$(extras))),)
    GROUP_FLAGS += '--group=test'
else
    GROUP_FLAGS += '--group=testqt'
endif

VERBOSE := $(if $(v),--verbose,)
ISOLATED := $(if $(isolated),--isolated)
PYTHON_FLAG := $(if $(py),-p=$(py))
RESOLUTION := $(if $(min),--resolution=lowest-direct)
COV := $(if $(cov),coverage run -p -m)

test:
	uv run $(VERBOSE) $(ISOLATED) $(PYTHON_FLAG) $(RESOLUTION) --exact --no-dev $(EXTRA_FLAGS) $(GROUP_FLAGS) $(COV) pytest $(VERBOSE) --color=yes

test-arrays:
	$(MAKE) test extras=pyqt,pygfx groups=array-libs

test-all:
	$(MAKE) test extras=pyqt,pygfx
	$(MAKE) test extras=pyqt,vispy
	$(MAKE) test extras=pyside,pygfx
	$(MAKE) test extras=jupyter,pygfx
	$(MAKE) test extras=wx,pygfx
	$(MAKE) test extras=wx,vispy
# $(MAKE) test extras=pyside,vispy
# $(MAKE) test extras=jupyter,vispy

docs:
	uv run --group docs mkdocs build --strict

docs-serve:
	uv run --group docs mkdocs serve --no-strict

lint: 
	uv run pre-commit run --all-files