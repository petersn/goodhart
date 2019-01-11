.PHONY: run
run:
	python ./ProxyUtility.py

.PHONY: plot
plot:
	python ./plot.py

.PHONY: clean
clean:
	rm -rf ./runs ./output.png
