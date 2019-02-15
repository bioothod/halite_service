#!/bin/bash

protoc -I proto/ --go_out=plugins=grpc:proto/ proto/halite_model.proto
protoc -I proto/config/ --go_out=plugins=grpc:proto/config/ proto/config/halite_service_config.proto proto/config/service_config.proto proto/config/session_manager.proto

#python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. halite_model.proto
