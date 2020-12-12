.PHONY: create_video test_model train_model

# Creates a recording of autonomous driving
recording:
	python ./tools/drive.py ./models/model.h5 /opt/run1
	
video:
	python ./tools/video.py /opt/run1 --fps 30
	
# Runs model to drive autonomously on the simulator
simulation:
	python ./tools/drive.py ./models/model.h5

# Train the model
training:
	python ./src/model.py

prediction:
	python ./src/predict.py

visualization:
	python ./src/plotter.py ./models/model.h5

# Set some git configurations. Only run on the workspace
gitconf:
	git config user.email "joustava@gmail.com"
	git config user.name "joustava"