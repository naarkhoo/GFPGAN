IMAGE_TAG := "gpgan_cpu"

build:
	docker buildx build --platform=linux/amd64 -t ${IMAGE_TAG} -f Dockerfile ../

run:
	# Make sure directories exist to prevent permission issues
	docker run \
		--platform linux/amd64 \
		-e AWS_PROFILE=$${AWS_PROFILE:-testing} \
		-e PYTHONUNBUFFERED=1 \
		-p 5000:5000 \
		--rm \
		-v `pwd`/../:/gfpgan/ \
		-v ~/.aws/:/root/.aws/:ro\
		${IMAGE_TAG}

interactive:
	docker run \
		--platform linux/amd64 \
		-it \
		-e AWS_PROFILE=$${AWS_PROFILE:-testing} \
		-e PYTHONUNBUFFERED=1 \
		-p 5000:5000 \
		--rm \
		-v `pwd`/../:/gfpgan/ \
		-v ~/.aws/:/root/.aws/:ro\
		${IMAGE_TAG} \
		/bin/bash
