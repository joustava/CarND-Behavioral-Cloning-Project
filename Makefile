.PHONY: create_video test_model train_model

recording:
		python ./tools/drive.py .models/model.h5 run1
		python ./tools/video.py run1 --fps 30
	
simulation:
		python ./tools/drive.py ./models/model.h5

training:
		python ./src/model.py