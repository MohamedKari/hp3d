all: docker

docker: protos
	DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker-compose up --build

docker-ssh:
	docker-compose run --entrypoint bash hp3d

protos: realtery-protos/spec/service/human_pose_detection_service.proto
	source env.sh && \
	source activate $$CONDA_ENV && \
	cp realtery-protos/spec/service/human_pose_detection_service.proto hp3d/rpc/ && \
	python -m grpc_tools.protoc \
		-I . \
		--python_out=. \
		--grpc_python_out=. \
		hp3d/rpc/*.proto