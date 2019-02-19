#!/bin/bash

buildenv --go --pull --image dockerio.badoo.com/rnd/buildenv-go-lite:release-20190203-tf-1.13-cuda-10.0 --go-pkg-mount rndgit.msk/goservice --go-pkg-mount github.com/golang --go-pkg-mount google.golang.org --go-pkg-mount golang.org/x go build
