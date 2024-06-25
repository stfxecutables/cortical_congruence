phony: all

all: html pdf

html:
	./build.sh

pdf:
	./build_pdf.sh

watch:
	watchexec --project-origin=. -w congruence.md -w mdstyle.html ./build.sh
