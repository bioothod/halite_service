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

	state_shape []int64
	params_shape []int64
	logits_shape []int64

	h *History

	train_step int

	saver_def []byte
	train_dir string
}

type BatchWrapper struct {
	batch []byte
}
func (b *BatchWrapper) Data() []byte {
	return b.batch
}

func (ctx *ServiceContext) GetFrozenGraph(_ctx context.Context, he *halite_proto.Status) (*halite_proto.FrozenGraph, error) {
	var b bytes.Buffer
	bwriter := bufio.NewWriter(&b)

	slot, err := ctx.sm.GetExecutionSlot(SERVICE_NAME, 1)
	if err != nil {
		return nil, fmt.Errorf("could not get execution slot: %v", err)
	}
	defer slot.Cleanup()

	_, err = slot.Graph().WriteTo(bwriter)
	if err != nil {
		return nil, fmt.Errorf("could not serialize graph_def: %v", err)
	}
	bwriter.Flush()

	prefix := fmt.Sprintf("%s/tmp_model.ckpt", ctx.train_dir)

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

	log.Infof("sending model: train_step: %d, graph_def: %d, checkpoint: %s, index: %d, data: %d",
		ctx.train_step, b.Len(), checkpoint, len(checkpoint_index), len(checkpoint_data))

	return &halite_proto.FrozenGraph {
		GraphDef: b.Bytes(),
		Prefix: prefix,
		SaverDef: ctx.saver_def,
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

func (ctx *ServiceContext) train() error {
	trlen := 40
	batch := ctx.h.Sample(trlen, 100)
	if len(batch) == 0 {
		log.Infof("there is no data, sleeping...")
		time.Sleep(1 * time.Second)
		return nil
	}

	log.Infof("sampled %d episodes", len(batch))

	for _, tr := range batch {
		input_tensors := make(map[string]*tf.Tensor)

		input_states := NewTensorSources(trlen)
		input_params := NewTensorSources(trlen)
		input_logits := make([][]float32, 0, trlen)
		input_actions := make([]int32, 0, trlen)
		input_rewards := make([]float32, 0, trlen)
		input_dones := make([]bool, 0, trlen)

		for _, e := range tr.Entries {
			sw := &BatchWrapper {
				batch: e.OldState.State,
			}
			input_states.Append(sw)

			sp := &BatchWrapper {
				batch: e.OldState.Params,
			}
			input_params.Append(sp)

			input_logits = append(input_logits, e.Logits)
			input_actions = append(input_actions, e.Action)
			input_rewards = append(input_rewards, e.Reward)
			input_dones = append(input_dones, e.Done)
		}

		var err error
		input_tensors["input/map"], err = tf.ReadTensor(tf.Float, append([]int64{int64(input_states.NumSources())}, ctx.state_shape...), input_states.NewReader())
		if err != nil {
			return fmt.Errorf("could not convert input state into tensor shape %v: %v", ctx.state_shape, err)
		}
		input_tensors["input/params"], err = tf.ReadTensor(tf.Float, append([]int64{int64(input_params.NumSources())}, ctx.params_shape...), input_params.NewReader())
		if err != nil {
			return fmt.Errorf("could not convert input params into tensor shape %v: %v", ctx.params_shape, err)
		}
		input_tensors["input/policy_logits"], err = tf.NewTensor(input_logits)
		if err != nil {
			return fmt.Errorf("could not convert input params into tensor shape %v: %v", ctx.logits_shape, err)
		}
		input_tensors["input/action_taken"], err = tf.NewTensor(input_actions)
		if err != nil {
			return fmt.Errorf("could not convert input actions into tensor: %v", err)
		}
		input_tensors["input/done"], err = tf.NewTensor(input_dones)
		if err != nil {
			return fmt.Errorf("could not convert input dones into tensor: %v", err)
		}
		input_tensors["input/reward"], err = tf.NewTensor(input_rewards)
		if err != nil {
			return fmt.Errorf("could not convert input rewards into tensor: %v", err)
		}
		input_tensors["learning_rate_ph"], err = tf.NewTensor(float32(0.001))

		slot, err := ctx.sm.GetExecutionSlot(SERVICE_NAME, 1)
		if err != nil {
			return fmt.Errorf("%s: could not get execution slot: %v", SERVICE_NAME, err)
		}
		defer slot.Cleanup()

		run := slot.NewSessionRun()

		run.AddOutput("output/policy_gradient_loss", DefaultConvert)
		run.AddOutput("output/baseline_loss", DefaultConvert)
		run.AddOutput("output/cross_entropy_loss", DefaultConvert)
		run.AddOutput("output/total_loss", DefaultConvert)
		run.AddTarget("output/train_op")

		err = run.Run(input_tensors)
		if err != nil {
			return fmt.Errorf("could not run inference: %v", err)
		}

		_policy_gradient_loss, err := run.Output("output/policy_gradient_loss")
		policy_gradient_loss := _policy_gradient_loss.(float32)
		_baseline_loss, err := run.Output("output/baseline_loss")
		baseline_loss := _baseline_loss.(float32)
		_cross_entropy_loss, err := run.Output("output/cross_entropy_loss")
		cross_entropy_loss := _cross_entropy_loss.(float32)
		_total_loss, err := run.Output("output/total_loss")
		total_loss := _total_loss.(float32)

		log.Infof("train step: %d, policy_gradient_loss: %.2e, baseline_loss: %.2e, cross_entropy_loss: %.2e, total_loss: %.2e",
			ctx.train_step, policy_gradient_loss, baseline_loss, cross_entropy_loss, total_loss)
		
		ctx.train_step += 1

		if ctx.train_step % 1000 == 0 {
			prefix := fmt.Sprintf("%s/model.ckpt-%d", ctx.train_dir, ctx.train_step)

			_, err = StoreVariablesIntoCheckpoint(slot.Graph(), slot.Session(), prefix)
			if err != nil {
				return fmt.Errorf("could not save checkpoint '%s': %v", prefix, err)
			}
		}
	}

	return nil
}

func (ctx *ServiceContext) start_training() {
	go func() {
		for {
			err := ctx.train()
			if err != nil {
				log.Fatalf("training failed: %v", err)
			}
		}
	}()
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

	tname = "input_policy_logits_shape"
	res, err = GetTensorByName(slot.Graph(), slot.Session(), tname)
	if err != nil {
		return fmt.Errorf("could not find tensor %s: %v", tname, err)
	}
	ctx.logits_shape = res[0].Value().([]int64)
	log.Infof("input: state shape: %v, params shape: %v, policy logits shape: %v", ctx.state_shape, ctx.params_shape, ctx.logits_shape)

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

	saver_def, err := ioutil.ReadFile(srv_config.GetSaverDef())
	if err != nil {
		log.Fatalf("could not read saver_def file '%s': %v", srv_config.GetSaverDef(), err)
	}

	log.Infof("saver def size: %d", len(saver_def))

	ctx := &ServiceContext {
		h: NewHistory(int(srv_config.GetMaxEpisodesPerClient()), int(srv_config.GetMaxEpisodesTotal())),
		saver_def : []byte(saver_def),
		train_dir: srv_config.GetTrainDir(),
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

	ctx.start_training()

	log.Infof("now serving on %s", srv_config.GetAddress())
	grpcServer.Serve(listener)

}
