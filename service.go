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
	"sync"
	"time"
)

type ServiceContext struct {
	sm  *SessionManager

	state_shape []int64
	params_shape []int64
	logits_shape []int64

	h *History

	train_step int
	saved_train_step int

	saver_def []byte
	train_dir string

	trlen int
	max_batch_size int

	checkpoint_steps int

	graph_buffer bytes.Buffer
	checkpoint_lock sync.Mutex

	batch_channel chan map[string]*tf.Tensor

	learning_rate float32

	service_name string

	c_state_size int
	h_state_size int
}

type BatchWrapper struct {
	batch []byte
}
func (b *BatchWrapper) Data() []byte {
	return b.batch
}

func (ctx *ServiceContext) gen_checkpoint_filenames() (string, string, string) {
	prefix := fmt.Sprintf("%s/tmp_model.ckpt", ctx.train_dir)
	index_file := fmt.Sprintf("%s.index", prefix)
	data_file := fmt.Sprintf("%s.data-00000-of-00001", prefix)

	return prefix, index_file, data_file
}

func (ctx *ServiceContext) cache_checkpoint() (int, error) {
	ctx.checkpoint_lock.Lock()
	defer ctx.checkpoint_lock.Unlock()

	if ctx.train_step != ctx.saved_train_step {
		log.Infof("caching checkpoint for train_step %d", ctx.train_step)

		prefix, _, _ := ctx.gen_checkpoint_filenames()

		slot, err := ctx.sm.GetExecutionSlot(ctx.service_name, 1)
		if err != nil {
			return -1, fmt.Errorf("could not get execution slot: %v", err)
		}
		defer slot.Cleanup()

		_, err = StoreVariablesIntoCheckpoint(slot.Graph(), slot.Session(), prefix)
		if err != nil {
			return -1, fmt.Errorf("could not save checkpoint '%s': %v", prefix, err)
		}

		ctx.saved_train_step = ctx.train_step
	}

	return ctx.saved_train_step, nil
}

func (ctx *ServiceContext) GetFrozenGraph(_ctx context.Context, he *halite_proto.Status) (*halite_proto.FrozenGraph, error) {
	train_step, err := ctx.cache_checkpoint()
	if err != nil {
		return nil, err
	}
	prefix, index_file, data_file := ctx.gen_checkpoint_filenames()

	checkpoint_index, err := ioutil.ReadFile(index_file)
	if err != nil {
		return nil, fmt.Errorf("could not read checkpoint index file '%s': %v", index_file, err)
	}
	checkpoint_data, err := ioutil.ReadFile(data_file)
	if err != nil {
		return nil, fmt.Errorf("could not read checkpoint data file '%s': %v", data_file, err)
	}

	log.Infof("sending model: train_step: %d, graph_def: %d, checkpoint: %s, index: %d, data: %d",
		train_step, ctx.graph_buffer.Len(), prefix, len(checkpoint_index), len(checkpoint_data))

	return &halite_proto.FrozenGraph {
		GraphDef: ctx.graph_buffer.Bytes(),
		Prefix: prefix,
		SaverDef: ctx.saver_def,
		CheckpointIndex: checkpoint_index,
		CheckpointData: checkpoint_data,
		TrainStep: int32(train_step),
		TrajectoryLen: int32(ctx.trlen),
	}, nil
}

func (ctx *ServiceContext) HistoryAppend(_ctx context.Context, htr *halite_proto.Trajectory) (*halite_proto.Status, error) {
	status := &halite_proto.Status{}

	if len(htr.CState) != ctx.c_state_size {
		return status, fmt.Errorf("invalid c_state size %d, must be %d", len(htr.CState), ctx.c_state_size)
	}
	if len(htr.HState) != ctx.h_state_size {
		return status, fmt.Errorf("invalid h_state size %d, must be %d", len(htr.HState), ctx.h_state_size)
	}
	if len(htr.Entries) != ctx.trlen {
		return status, fmt.Errorf("invalid trajectory len %d, must be %d", len(htr.Entries), ctx.trlen)
	}

	ctx.h.AddTrajectory(htr)

	return status, nil
}

func (ctx *ServiceContext) generate_batch() error {
	start_time := time.Now()

	batch, input_c_states, input_h_states := ctx.h.Sample(ctx.trlen, ctx.max_batch_size)
	if batch == nil || len(batch) != ctx.max_batch_size {
		time.Sleep(100 * time.Millisecond)
		return nil
	}

	batch_sampling_time_ms := time.Since(start_time).Seconds() * 1000
	tensors_start_time := time.Now()

	input_tensors := make(map[string]*tf.Tensor)

	input_states := NewTensorSources(ctx.trlen * len(batch))
	input_params := NewTensorSources(ctx.trlen * len(batch))
	input_logits := make([][]float32, 0, ctx.trlen * len(batch))
	input_actions := make([]int32, 0, ctx.trlen * len(batch))
	input_rewards := make([]float32, 0, ctx.trlen * len(batch))
	input_dones := make([]bool, 0, ctx.trlen * len(batch))

	for _, trj := range batch {
		for _, e := range trj {
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
	input_tensors["input/time_steps"], err = tf.NewTensor(int32(ctx.trlen))
	if err != nil {
		return fmt.Errorf("could not convert input time steps into tensor: %v", err)
	}
	input_tensors["learning_rate_ph"], err = tf.NewTensor(ctx.learning_rate)
	if err != nil {
		return fmt.Errorf("could not convert learning rate into tensor: %v", err)
	}

	input_tensors["impala_rnn/input/c_state"], err = tf.NewTensor(input_c_states)
	if err != nil {
		return fmt.Errorf("could not convert input c_states into tensor: %v", err)
	}
	input_tensors["impala_rnn/input/h_state"], err = tf.NewTensor(input_h_states)
	if err != nil {
		return fmt.Errorf("could not convert input h_states into tensor: %v", err)
	}

	train_preparation_time_ms := time.Since(tensors_start_time).Seconds() * 1000

	ctx.h.Lock()
	num_trajectories := ctx.h.NumTrajectories
	num_clients := len(ctx.h.Clients)
	ctx.h.Unlock()

	log.Infof("%d: trjs: %d, batch_sampling: %.1f ms, train_preparation: %.1f ms, total trajectories: %d, total clients: %d",
			ctx.train_step, len(batch), batch_sampling_time_ms, train_preparation_time_ms,
			num_trajectories, num_clients)

	ctx.batch_channel <- input_tensors
	return nil
}

func (ctx *ServiceContext) train() error {
	slot, err := ctx.sm.GetExecutionSlot(ctx.service_name, 1)
	if err != nil {
		return fmt.Errorf("%s: could not get execution slot: %v", ctx.service_name, err)
	}
	defer slot.Cleanup()

	run := slot.NewSessionRun()

	run.AddOutput("output/policy_gradient_loss")
	run.AddOutput("output/baseline_loss")
	run.AddOutput("output/cross_entropy_loss")
	run.AddOutput("output/total_loss")
	run.AddTarget("output/train_op")

	for {
		start_time := time.Now()

		input_tensors := <-ctx.batch_channel

		tensors_waiting_time_ms := time.Since(start_time).Seconds() * 1000

		train_start_time := time.Now()
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

		train_time_ms := time.Since(train_start_time).Seconds() * 1000

		log.Infof("%d: trjs: %d, policy_gradient_loss: %.2e, baseline_loss: %.2e, cross_entropy_loss: %.2e, total_loss: %.2e, " +
			"tensor waiting: %.1f ms, train: %.1f ms",
				ctx.train_step, input_tensors["input/map"].Shape()[0] / int64(ctx.trlen),
				policy_gradient_loss, baseline_loss, cross_entropy_loss, total_loss,
				tensors_waiting_time_ms, train_time_ms)

		ctx.checkpoint_lock.Lock()
		ctx.train_step += 1
		ctx.checkpoint_lock.Unlock()

		if ctx.train_step % ctx.checkpoint_steps == 0 {
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
			err := ctx.generate_batch()
			if err != nil {
				log.Fatalf("could not generate batch: %v", err)
			}


		}
	}()
	go func() {
		for {
			err := ctx.train()
			if err != nil {
				log.Fatalf("training has failed: %v", err)
			}
		}
	}()
}

func (ctx *ServiceContext) FillConfig() error {
	slot, err := ctx.sm.GetExecutionSlot(ctx.service_name, 1)
	if err != nil {
		return fmt.Errorf("%s: could not get execution slot: %v", ctx.service_name, err)
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

	tname = "impala_rnn/output/lstm_state_sizes"
	res, err = GetTensorByName(slot.Graph(), slot.Session(), tname)
	if err != nil {
		return fmt.Errorf("could not find tensor %s: %v", tname, err)
	}
	state_sizes := res[0].Value().([]int32)

	ctx.c_state_size = int(state_sizes[0])
	ctx.h_state_size = int(state_sizes[1])

	log.Infof("input: state shape: %v, params shape: %v, policy logits shape: %v, c_state_size: %d, h_state_size: %d",
		ctx.state_shape, ctx.params_shape, ctx.logits_shape, ctx.c_state_size, ctx.h_state_size)

	bwriter := bufio.NewWriter(&ctx.graph_buffer)
	_, err = slot.Graph().WriteTo(bwriter)
	if err != nil {
		return fmt.Errorf("could not serialize graph_def: %v", err)
	}
	bwriter.Flush()

	return nil
}

func main() {
	cfg_path := flag.String("c", "halite_service.conf", "Halite service config file")
	cpu_only := flag.Bool("cpu_only", false, "if specified, only CPU devices will be used if specified in service configs")
	gpu_only := flag.Bool("gpu_only", false, "if specified, only GPU devices will be used if specified in service configs")
	service_name := flag.String("service_name", "", "if specified, use this default name for the service")
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

	prune_timeout := time.Duration(srv_config.GetPruneOldClientsTimeoutSeconds()) * time.Second

	ctx := &ServiceContext {
		saver_def : []byte(saver_def),
		train_dir: srv_config.GetTrainDir(),
		trlen: int(srv_config.GetTrajectoryLen()),
		max_batch_size: int(srv_config.GetMaxBatchSize()),
		checkpoint_steps: int(srv_config.GetCheckpointSteps()),
		batch_channel: make(chan map[string]*tf.Tensor, int(srv_config.GetTrajectoryChannelSize())),
		learning_rate: srv_config.GetLearningRate(),
		service_name: *service_name,
		saved_train_step: -1,
		train_step: 0,
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

	ctx.h = NewHistory(int(srv_config.GetMaxTrajectoriesPerClient()), prune_timeout)

	var opts []grpc.ServerOption
	opts = append(opts, grpc.MaxRecvMsgSize(100*1024*1024))
	opts = append(opts, grpc.MaxSendMsgSize(100*1024*1024))

	grpcServer := grpc.NewServer(opts...)
	halite_proto.RegisterHaliteProcessServer(grpcServer, ctx)

	ctx.start_training()

	log.Infof("now serving on %s", srv_config.GetAddress())
	grpcServer.Serve(listener)

}
