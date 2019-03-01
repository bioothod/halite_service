package main

import (
	"bufio"
	"bytes"
	"context"
	"flag"
	"fmt"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/bioothod/halite/proto"
	"github.com/bioothod/halite/proto/config"
	"google.golang.org/grpc"
	//"google.golang.org/grpc/codes"
	//"google.golang.org/grpc/status"
	"io/ioutil"
	"math/rand"
	"net"
	"rndgit.msk/goservice/log"
	"time"
)

const SERVICE_NAME = "halite_service"

type ServiceContext struct {
	sm  *SessionManager

	explore_eps float32

	state_shape []int64
	params_shape []int64

	h *History
}

type BatchWrapper struct {
	batch []byte
}
func (b *BatchWrapper) Data() []byte {
	return b.batch
}

func (ctx *ServiceContext) qvals_to_tensor(batch *halite_proto.StateBatch) (*ExecutionSlot, map[string]*tf.Tensor, error) {
	input_tensors := make(map[string]*tf.Tensor)

	input_states := NewTensorSources(len(batch.Batch))
	input_params := NewTensorSources(len(batch.Batch))

	for _, st := range batch.Batch {
		sw := &BatchWrapper {
			batch: st.GetState(),
		}
		input_states.Append(sw)

		sp := &BatchWrapper {
			batch: st.GetParams(),
		}
		input_params.Append(sp)
	}

	var err error
	input_tensors["input_states"], err = tf.ReadTensor(tf.Float, append([]int64{int64(input_states.NumSources())}, ctx.state_shape...), input_states.NewReader())
	if err != nil {
		return nil, nil, fmt.Errorf("could not convert input state into tensor shape %v: %v", ctx.state_shape, err)
	}
	input_tensors["input_params"], err = tf.ReadTensor(tf.Float, append([]int64{int64(input_params.NumSources())}, ctx.params_shape...), input_params.NewReader())
	if err != nil {
		return nil, nil, fmt.Errorf("could not convert input params into tensor shape %v: %v", ctx.params_shape, err)
	}
	input_tensors["explore_eps"], err = tf.NewTensor(ctx.explore_eps)

	slot, err := ctx.sm.GetExecutionSlot(SERVICE_NAME, 1)
	if err != nil {
		return nil, nil, fmt.Errorf("%s: could not get execution slot: %v", SERVICE_NAME, err)
	}
	return slot, input_tensors, nil
}

func (ctx *ServiceContext) GetFrozenGraph(_ctx context.Context, he *halite_proto.Status) (*halite_proto.FrozenGraph, error) {
	var b bytes.Buffer
	bwriter := bufio.NewWriter(&b)

	slot, err := ctx.sm.GetExecutionSlot("", 1)
	if err != nil {
		return nil, fmt.Errorf("could not get execution slot: %v", err)
	}
	defer slot.Cleanup()

	_, err = slot.Graph().WriteTo(bwriter)
	if err != nil {
		return nil, fmt.Errorf("could not serialize graph_def: %v", err)
	}
	bwriter.Flush()

	prefix := "model.ckpt"

	checkpoint, err := StoreVariablesIntoCheckpoint(slot.Graph(), slot.Session(), prefix)
	if err != nil {
		return nil, fmt.Errorf("could not save checkpoint '%s': %v", prefix, err)
	}

	index_file := fmt.Sprintf("%s.index", checkpoint)
	checkpoint_index, err := ioutil.ReadFile(index_file)
	if err != nil {
		return nil, fmt.Errorf("could not read checkpoint index file '%s': %v", index_file, err)
	}
	data_file := fmt.Sprintf("%s.data-00000-of-00001", checkpoint)
	checkpoint_data, err := ioutil.ReadFile(data_file)
	if err != nil {
		return nil, fmt.Errorf("could not read checkpoint data file '%s': %v", data_file, err)
	}
	saver_def_file := fmt.Sprintf("saver_def.pb")
	saver_def, err := ioutil.ReadFile(saver_def_file)
	if err != nil {
		return nil, fmt.Errorf("could not read saver_def file '%s': %v", saver_def_file, err)
	}

	log.Infof("saved model: graph_def: %d, checkpoint: %s, index: %d, data: %d, saver_def: %d", b.Len(), checkpoint, len(checkpoint_index), len(checkpoint_data), len(saver_def))
	return &halite_proto.FrozenGraph {
		GraphDef: b.Bytes(),
		Prefix: prefix,
		SaverDef: saver_def,
		CheckpointIndex: checkpoint_index,
		CheckpointData: checkpoint_data,
	}, nil
}

func (ctx *ServiceContext) HistoryAppend(_ctx context.Context, he *halite_proto.HistoryEntry) (*halite_proto.Status, error) {
	ctx.h.Append(he)

	status := &halite_proto.Status{}
	return status, nil
}

func (ctx *ServiceContext) TrainStep(_ctx context.Context, _ *halite_proto.Status) (*halite_proto.Status, error) {
	status := &halite_proto.Status{}
	return status, nil
}

func (ctx *ServiceContext) FillConfig() error {
	slot, err := ctx.sm.GetExecutionSlot(SERVICE_NAME, 1)
	if err != nil {
		return fmt.Errorf("%s: could not get execution slot: %v", SERVICE_NAME, err)
	}
	defer slot.Cleanup()

	tname := "input_map_shape"
	res, err := GetTensorByName(slot.Graph(), slot.Session(), tname)
	if err != nil {
		return fmt.Errorf("could not find tensor %s: %v", tname, err)
	}
	ctx.state_shape = res[0].Value().([]int64)

	tname = "input_params_shape"
	res, err = GetTensorByName(slot.Graph(), slot.Session(), tname)
	if err != nil {
		return fmt.Errorf("could not find tensor %s: %v", tname, err)
	}
	ctx.params_shape = res[0].Value().([]int64)

	log.Infof("input: state shape: %v, params shape: %v", ctx.state_shape, ctx.params_shape)

	return nil
}

func main() {
	cfg_path := flag.String("c", "halite_service.conf", "Halite service config file")
	cpu_only := flag.Bool("cpu_only", false, "if specified, only CPU devices will be used if specified in service configs")
	gpu_only := flag.Bool("gpu_only", false, "if specified, only GPU devices will be used if specified in service configs")
	flag.Parse()

	config := &halite_config.HaliteConfig{}
	err := ParseConfigFromFile(*cfg_path, config)
	if err != nil {
		log.Fatalf("could not unmarshal config from %s: %v", *cfg_path, err)
	}

	srv_config := config.GetServiceConfig()

	listener, err := net.Listen("tcp", srv_config.GetAddress())
	if err != nil {
		log.Fatalf("could not start listener on %s: %v", srv_config.GetAddress(), err)
	}

	rand.Seed(time.Now().UnixNano())

	ctx := &ServiceContext {
		h: NewHistory(int(srv_config.GetMaxEpisodesPerClient()), int(srv_config.GetMaxEpisodesTotal())),
		explore_eps: srv_config.GetExploreEps(),
	}

	ctx.sm, err = NewSessionManagerFromConfigWithWildcards(config.GetSessionManagerConfig(), *cpu_only, *gpu_only)
	if err != nil {
		log.Fatalf("could not create new session manager from config: %v: %v", config.GetSessionManagerConfig(), err)
	}
	defer ctx.sm.Cleanup()

	err = ctx.FillConfig()
	if err != nil {
		log.Fatalf("could not read config tensors: %v", err)
	}

	grpcServer := grpc.NewServer()
	halite_proto.RegisterHaliteProcessServer(grpcServer, ctx)

	log.Infof("now serving on %s", srv_config.GetAddress())
	grpcServer.Serve(listener)

}
