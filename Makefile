all: docker

docker:
	DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker-compose up --build

docker-ssh:
	# docker run -it --entrypoint bash hp3d
	docker-compose run --entrypoint bash hp3d