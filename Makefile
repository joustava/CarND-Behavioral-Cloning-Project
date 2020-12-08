.PHONY: create_video test_model train_model

create_video:
		python ./tools/drive.py .models/model.h5 run1
		python ./tools/video.py run1 --fps 30
	
test_model:
		python ./tools/drive.py ./models/model.h5

train_model:
		python ./src/model.py