clean:
	rm -r __pycache__
	rm -r */__pycache__

clean-saves:
	rm -r logs/*
	rm -rf modeling/save/h_*

clean-logs:
	rm -r logs/*

clean-save:
	rm -rf modeling/save/h_*

clean-result:
	rm result/*

clean-vector:
	rm -r modeling/vectors/vectors_*
