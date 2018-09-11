clean:
	rm -r __pycache__
	rm -r */__pycache__

clean-saves:
	rm -r logs/log_*
	rm -rf modeling/save/h_*

clean-log:
	rm -r logs/log_*

clean-save:
	rm -rf modeling/save/h_*

clean-result:
	rm result/*

clean-vector:
	rm -r modeling/vectors/vectors_*
