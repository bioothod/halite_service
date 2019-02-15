package main

import (
	"io"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"sync"
)

type TensorSource interface {
	Data() []byte
}

type TensorSources struct {
	wg sync.WaitGroup
	mtx sync.Mutex

	sources []TensorSource
}

func NewTensorSources(num int) *TensorSources {
	sources := &TensorSources {
		sources: make([]TensorSource, 0),
	}

	sources.wg.Add(num)

	return sources
}

func (sources *TensorSources) Append(src TensorSource) {
	sources.mtx.Lock()
	defer sources.mtx.Unlock()

	sources.sources = append(sources.sources, src)
}

func (sources *TensorSources) NumSources() int {
	return len(sources.sources)
}

func (sources *TensorSources) Wait() {
	sources.wg.Wait()
}

func (sources *TensorSources) Done() {
	sources.wg.Done()
}

func (sources *TensorSources) GetSource(i int) TensorSource {
	return sources.sources[i]
}

type TensorSourcesReader struct {
	sources *TensorSources
	image_index int
	within_image_offset int
}

func (sources *TensorSources) NewReader() io.Reader {
	return &TensorSourcesReader {
		sources: sources,
		image_index: 0,
		within_image_offset: 0,
	}
}

func (r *TensorSourcesReader) Read(dst []byte) (n int, err error) {
	read_size := 0

	dst_offset := 0
	dst_avail := len(dst)

	for read_size < len(dst) {
		if r.image_index == r.sources.NumSources() {
			break
		}

		img := r.sources.sources[r.image_index]
		src := img.Data()

		avail := len(src) - r.within_image_offset
		if avail <= 0 {
			r.image_index += 1
			r.within_image_offset = 0
			continue
		}

		if avail > dst_avail {
			avail = dst_avail
		}

		num := copy(dst[dst_offset:], src[r.within_image_offset : r.within_image_offset + avail])
		if num > 0 {
			r.within_image_offset += num
			read_size += num
			dst_avail -= num
			dst_offset += num
		}

		if num == 0 {
			return read_size, io.EOF
		}
	}

	if read_size == 0 {
		return read_size, io.EOF
	}

	return read_size, nil
}

func (sources *TensorSources) ToTensor(input_shape []int64) (*tf.Tensor, error) {
	return tf.ReadTensor(tf.Uint8, append([]int64{int64(sources.NumSources())}, input_shape...), sources.NewReader())
}
