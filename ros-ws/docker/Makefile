.PHONY: up shell build build-all test down logs

# Parallel packages x make jobs; keep peak RAM in check (raise on bigger machines).
CATKIN_PKGS ?= 2
CATKIN_JOBS ?= 3
CATKIN_CONFIG := catkin config --extend /py3_ws/devel --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3
# The container runs as root; hand build outputs back to the host user.
CHOWN := chown -R $(shell id -u):$(shell id -g) build devel logs .catkin_tools

up:
	docker compose up --build --detach

shell:
	xhost +local: && docker compose exec dev bash

# Project packages only (the PAL stack and Python 3 overlay are prebuilt in the image).
build:
	docker compose exec dev bash -lc "$(CATKIN_CONFIG) && catkin build -p $(CATKIN_PKGS) -j $(CATKIN_JOBS); \
		rc=\$$?; $(CHOWN); exit \$$rc"

build-all:
	docker compose exec dev bash -lc "catkin clean --yes && $(CATKIN_CONFIG) && \
		catkin build -p $(CATKIN_PKGS) -j $(CATKIN_JOBS); rc=\$$?; $(CHOWN); exit \$$rc"

test:
	docker compose exec dev bash -lc "python3 -m pytest -q -p no:cacheprovider src/tiago/tiago_door_planning/tests"

down:
	docker compose down

logs:
	docker compose logs --follow
