package main

import (
	"container/list"
	"fmt"
	"github.com/bioothod/halite/proto/config"
	"github.com/gogo/protobuf/proto"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	tensorflow_config "github.com/tensorflow/tensorflow/tensorflow/go/core/protobuf"
	"io/ioutil"
	"math"
	"os"
	"rndgit.msk/goservice/log"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"time"
)

type ConversionFunction func(input interface{}) (interface{}, error)
func DefaultConvert(input interface{}) (interface{}, error) {
	return input, nil
}

func ConvertToArrayOfFloat32(input interface{}) (interface{}, error) {
	x, ok := input.([][]float32)
	if !ok {
		return nil, fmt.Errorf("[][]float32 type mismatch for input %v with type %v", input, reflect.TypeOf(input))
	}
	return x, nil
}
func ConvertToArrayOfInt32(input interface{}) (interface{}, error) {
	x, ok := input.([][]int32)
	if !ok {
		return nil, fmt.Errorf("[][]float32 type mismatch for input %v with type %v", input, reflect.TypeOf(input))
	}
	return x, nil
}
func ConvertToInt64(input interface{}) (interface{}, error) {
	x, ok := input.([]int64)
	if !ok {
		return nil, fmt.Errorf("[]int64 type mismatch for input %v with type %v", input, reflect.TypeOf(input))
	}
	return x, nil
}
func ConvertToInt32(input interface{}) (interface{}, error) {
	x, ok := input.([]int32)
	if !ok {
		return nil, fmt.Errorf("[]int32 type mismatch for input %v with type %v", input, reflect.TypeOf(input))
	}
	return x, nil
}
func ConvertToString(input interface{}) (interface{}, error) {
	x, ok := input.([]string)
	if !ok {
		return nil, fmt.Errorf("[]string type mismatch for input %v with type %v", input, reflect.TypeOf(input))
	}
	return x, nil
}
func ConvertToBool(input interface{}) (interface{}, error) {
	x, ok := input.([]bool)
	if !ok {
		return nil, fmt.Errorf("[]bool type mismatch for input %v with type %v", input, reflect.TypeOf(input))
	}
	return x, nil
}
func ConvertToFloat32(input interface{}) (interface{}, error) {
	x, ok := input.([]float32)
	if !ok {
		return nil, fmt.Errorf("[]float32 type mismatch for input %v with type %v", input, reflect.TypeOf(input))
	}
	return x, nil
}

type SessionRun struct {
	slot *ExecutionSlot
	output_operations map[string]*OutputOperation
	outputs []tf.Output
}

type OutputOperation struct {
	Name string
	Output tf.Output

	Convert ConversionFunction

	Index int
	Result interface{}
}

type ServiceModel struct {
	graph *tf.Graph
	session *tf.Session
	outputs []string
}

func (srv *ServiceModel) Cleanup() {
	srv.session.Close()
}

type DeviceConfig struct {
	Priority int
	MaxSlots int
}

func (conf *DeviceConfig) need_setup() bool {
	if conf.Priority == 0 || conf.MaxSlots == 0 {
		return true
	}

	return false
}

type Device struct {
	config          DeviceConfig
	slots_available int

	services map[string]*ServiceModel
}

func NewDevice(conf DeviceConfig) *Device {
	return &Device{
		config:          conf,
		slots_available: conf.MaxSlots,
		services:        make(map[string]*ServiceModel),
	}
}

func (dev *Device) Cleanup() {
	for _, srv := range dev.services {
		srv.Cleanup()
	}

	dev.services = make(map[string]*ServiceModel)
}

func (dev *Device) AddService(name string, service *ServiceModel) {
	dev.services[name] = service
}

type ServingDevices struct {
	service_name string
	slots_available int

	devices []*Device

	input_shapes map[string][]int64

	queue *list.List
}

type SessionManager struct {
	// device name -> Device
	devices map[string]*Device

	// service name -> list of devices, which serve this service
	serving_devices map[string]*ServingDevices

	slots_available_from_config int // max number of parallel requests specified in config
	slots_available int
	lock            *sync.Mutex

	graph   *tf.Graph
	session *tf.Session

	// when asking for execution slot without service name, use this one
	default_service_name string

	default_deadline_timeout time.Duration
	max_batch_size int

	cpu_only, gpu_only bool
}

func (ctx *SessionManager) DefaultServiceName() string {
	return ctx.default_service_name
}

func (ctx *SessionManager) DefaultDeadlineTimeout() time.Duration {
	return ctx.default_deadline_timeout
}

func (ctx *SessionManager) MaxBatchSize() int {
	return ctx.max_batch_size
}

func (ctx *SessionManager) Services() []string {
	var services []string
	for service_name, _ := range ctx.serving_devices {
		services = append(services, service_name)
	}

	return services
}

func (ctx *SessionManager) session_config(num_threads int32) *tf.SessionOptions {
	config := &tensorflow_config.ConfigProto{
		AllowSoftPlacement: true,
		GpuOptions:         &tensorflow_config.GPUOptions{},
	}

	if num_threads == 0 {
		log.Infof("using a single thread pool with the default amount of threads")
	} else {
		log.Infof("using a single thread pool with %d threads", num_threads)
	}

	config.SessionInterOpThreadPool = []*tensorflow_config.ThreadPoolOptionProto{{
		NumThreads: num_threads,
		GlobalName: "SinglePool",
	}}

	ser, err := proto.Marshal(config)
	if err != nil {
		log.Fatalf("could not serialize config: %v", err)
	}

	return &tf.SessionOptions{
		Config: ser,
	}
}

func NewSessionManager(max_parallel_requests_total int, devices map[string]DeviceConfig) (*SessionManager, error) {
	log.Infof("libtensorflow: %v, api version: %v, fetching devices...", tf.Version(), tf.APIVersion())
	if err := os.Setenv("TF_CPP_MIN_LOG_LEVEL", "1"); err != nil {
		return nil, fmt.Errorf("os.Setenv() failed: %v", err)
	}

	lock := &sync.Mutex{}
	ctx := &SessionManager{
		devices:         make(map[string]*Device),
		serving_devices: make(map[string]*ServingDevices),
		slots_available: 0,
		slots_available_from_config: max_parallel_requests_total,
		lock:            lock,
	}

	var err error
	ctx.graph = tf.NewGraph()
	ctx.session, err = tf.NewSession(ctx.graph, nil)
	if err != nil {
		return nil, fmt.Errorf("could not create default session: %v", err)
	}

	device_names, err := ctx.session.ListDevices()
	if err != nil {
		ctx.Cleanup()
		return nil, fmt.Errorf("could not list devices: %v", err)
	}

	for i, d := range device_names {
		name := d.Name
		parts := strings.Split(name, "/")
		if len(parts) > 0 {
			name = fmt.Sprintf("/%s", parts[len(parts)-1])
		}

		conf, ok := devices[name]
		if !ok {
			wname := fmt.Sprintf("/device:%s:*", d.Type)
			wconf, ok := devices[wname]
			if !ok {
				log.Errorf("hardware device %s does not match anything in config", name)
				continue
			}

			conf = wconf
			log.Infof("hardware device %s: using wildcard config for %s", name, wname)
		}

		log.Infof("hardware device %v: name: %v -> %s, type: %v, memory_limit_bytes: %v, priority: %d, execution slots: %d",
			i, d.Name, name, d.Type, d.MemoryLimitBytes, conf.Priority, conf.MaxSlots)


		dev := NewDevice(conf)
		ctx.devices[name] = dev
	}

	return ctx, nil
}

func NewSessionManagerFromConfigWithWildcards(config *halite_config.SessionManagerConfig, cpu_only, gpu_only bool) (*SessionManager, error) {
	max_parallel_requests_total := int(config.GetParallelSlotsTotal())

	if max_parallel_requests_total <= 0 {
		max_parallel_requests_total = runtime.NumCPU()
		log.Infof("there is no parallel_slots_total, using default: %d", max_parallel_requests_total)
	}

	dev_configs := make(map[string]DeviceConfig)
	dconf := config.GetDeviceConfig()
	for _, hw_dev := range dconf {
		name := hw_dev.GetName()
		max_slots := int(hw_dev.GetMaxSlots())
		if max_slots == 0 {
			log.Errorf("config: device: %s: no MaxSlots option, skipping", name)
			continue
		}

		priority := int(hw_dev.GetPriority())
		if priority == 0 {
			log.Errorf("config: device: %s: no Priority option, skipping", name)
			continue
		}

		conf := DeviceConfig{
			Priority: priority,
			MaxSlots: max_slots,
		}
		dev_configs[name] = conf

		log.Infof("config: device %s: %+v", name, conf)
	}

	sm, err := NewSessionManager(max_parallel_requests_total, dev_configs)
	if err != nil {
		return nil, err
	}

	num_threads := config.GetNumThreads()

	sm.cpu_only = cpu_only
	sm.gpu_only = gpu_only

	sm.default_service_name = config.GetDefaultServiceName()
	sm.max_batch_size = int(config.GetMaxBatchSize())
	sm.default_deadline_timeout = time.Duration(config.GetDefaultDeadlineTimeoutMs()) * time.Millisecond

	for _, srv := range config.GetServices() {
		srv_name := srv.GetName()
		model_file := srv.GetModel()
		srv_devs := srv.GetDevices()
		inputs := srv.GetInputs()
		checkpoint := srv.GetCheckpoint()

		if len(sm.default_service_name) == 0 {
			sm.default_service_name = srv_name
		}

		err = sm.BindService(srv_name, srv_devs, model_file, inputs, num_threads, checkpoint)
		if err != nil {
			log.Fatalf("%s: requested devices: %v, model file: %v, checkpoint: %s, could not bind service: %v", srv_name, srv_devs, model_file, checkpoint, err)
		}
	}

	log.Infof("using default service name \"%s\"", sm.default_service_name)

	return sm, nil
}

func check_dims(dims int64, channels int64) []int64 {
	remainder := math.Mod(float64(dims), float64(channels))
	if remainder != 0 {
		return nil
	}

	wh := dims / channels
	dim := math.Sqrt(float64(wh))

	if dim == math.Ceil(dim) && dim == math.Floor(dim) {
		return []int64{int64(dim), int64(dim), channels}
	}

	return nil
}

func get_output_shapes(op *tf.Operation) ([][]int64, error) {
	var out tf.Output
	var shapes [][]int64
	for i := 0; i < op.NumOutputs(); i += 1 {
		out = op.Output(i)

		shape, err := out.Shape().ToSlice()
		if err != nil {
			return nil, fmt.Errorf("operation: %s, type: %v, output: %d/%d, shape: %v, could not convert to slice: %v",
				op.Name(), op.Type(), i, op.NumOutputs(), out.Shape(), err)
		}

		shapes = append(shapes, shape)
	}

	return shapes, nil
}

func get_shape_by_reshape(op *tf.Operation) []int64 {
	var out tf.Output
	for i := 0; i < op.NumOutputs(); i += 1 {
		out = op.Output(i)

		for _, consumer := range out.Consumers() {
			log.Infof("input operation: %s, type: %v, output: %d/%d, consumer: %s, type: %s",
				op.Name(), op.Type(), i, op.NumOutputs(), consumer.Op.Name(), consumer.Op.Type())

			if consumer.Op.Type() == "Reshape" {
				reshape_shapes, err := get_output_shapes(consumer.Op)
				if err != nil {
					log.Infof("input operation: %s, type: %v, output: %d/%d, consumer: %s, type: %s: could not get output shape: %v",
						op.Name(), op.Type(), i, op.NumOutputs(), consumer.Op.Name(), consumer.Op.Type(), err)
					continue
				}

				for _, shape := range reshape_shapes {
					if len(shape) == 4 {
						return shape[1:]
					}
				}
			}
		}
	}

	return nil
}

func execute_array_tensor(g *tf.Graph, s *tf.Session, name string) ([]string, error) {
	res, err := GetTensorByName(g, s, name)
	if err != nil {
		return nil, err
	}

	arr, ok := res[0].Value().([]string)
	log.Infof("%s: %v, conversion ok: %v", name, arr, ok)

	if !ok {
		return nil, fmt.Errorf("%s: is not array of strings: %v", name, res[0].Value())
	}

	return arr, nil
}

func GetTensorByName(g *tf.Graph, s *tf.Session, name string) ([]*tf.Tensor, error) {
	tensor := g.Operation(name)
	if tensor == nil {
		return nil, fmt.Errorf("there is no tensor %s", name)
	}

	res, err := s.Run(map[tf.Output]*tf.Tensor{}, []tf.Output{tensor.Output(0)}, nil)
	if err != nil {
		return nil, fmt.Errorf("%s: can not execute: %v", name, err)
	}

	return res, nil
}

func try_to_find_input_output_tensors(g *tf.Graph, s *tf.Session) ([]string, []string, error) {
	inputs, err := execute_array_tensor(g, s, "input_names_var")
	if err != nil {
		return nil, nil, err
	}

	outputs, err := execute_array_tensor(g, s, "output_names_var")
	if err != nil {
		return nil, nil, err
	}

	return inputs, outputs, nil
}

func (ctx *SessionManager) CheckWildcardDevices(devices []string) []string {
	new_devices := make([]string, 0)
	for _, dev_name := range devices {
		if ctx.cpu_only && !strings.Contains(dev_name, "/device:CPU:") {
			continue
		}
		if ctx.gpu_only && !strings.Contains(dev_name, "/device:GPU:") {
			continue
		}

		if dev_name[len(dev_name) - 1] != '*' {
			new_devices = append(new_devices, dev_name)
			continue
		}

		prefix := dev_name[0 : len(dev_name) - 1]

		for physical_name, _ := range ctx.devices {
			if strings.HasPrefix(physical_name, prefix) {
				new_devices = append(new_devices, physical_name)
			}
		}
	}

	return new_devices
}

func StoreVariablesIntoCheckpoint(graph *tf.Graph, sess *tf.Session, prefix string) (string, error) {
	t, err := tf.NewTensor(prefix)
	if err != nil {
		return "", err
	}
	o := graph.Operation("save/Const").Output(0)
	ret, err := sess.Run(map[tf.Output]*tf.Tensor{o: t}, []tf.Output{graph.Operation("save/control_dependency").Output(0)}, nil)
	if err != nil {
		return "", err
	}
	return ret[0].Value().(string), nil
}

func restore_variables_from_checkpoint(graph *tf.Graph, sess *tf.Session, path string) error {
	t, err := tf.NewTensor(path)
	if err != nil {
		return err
	}
	o := graph.Operation("save/Const").Output(0)
	_, err = sess.Run(map[tf.Output]*tf.Tensor{o: t}, nil, []*tf.Operation{graph.Operation("save/restore_all")})
	return err
}

// New service can be only added before requesting any excution slot
func (ctx *SessionManager) BindService(name string, devices []string, model_file string, inputs []string, num_threads int32, checkpoint string) error {
	model, err := ioutil.ReadFile(model_file)
	if err != nil {
		return fmt.Errorf("could not read model file %s: %v", model_file, err)
	}

	if len(devices) == 0 {
		devices = append(devices, "/device:CPU:0")
	}

	devices = ctx.CheckWildcardDevices(devices)

	serving_devices := make([]*Device, 0)
	serving_slots := 0

	input_shapes := make(map[string][]int64)

	bound_devices := make([]string, 0)
	for _, dev_name := range devices {
		dev, ok := ctx.devices[dev_name]
		if !ok {
			log.Errorf("%s: device does not exist, can not bind model %s", dev_name, model_file)
			continue
		}

		opts := tf.GraphImportOptions{
			Prefix: "",
			Device: dev_name,
		}

		g := tf.NewGraph()
		if err := g.ImportWithOptions(model, opts); err != nil {
			return fmt.Errorf("%s: could not import graph from %s and device %s: %v", dev_name, model_file, opts.Device, err)
		}

		so := ctx.session_config(num_threads)
		session, err := tf.NewSession(g, so)
		if err != nil {
			return fmt.Errorf("%s: could not create session: %v", dev_name, err)
		}

		if checkpoint != "" {
			err = restore_variables_from_checkpoint(g, session, checkpoint)
			if err != nil {
				return fmt.Errorf("%s: checkpoint: %s: could not restore variables: %v", dev_name, checkpoint, err)
			}

			log.Infof("%s: successfully restored variables from checkpoint %s", dev_name, checkpoint)
		}

		inputs_from_graph, outputs_from_graph, err := try_to_find_input_output_tensors(g, session)
		log.Infof("tried to parse graph: inputs: %v, output: %v, err: %v", inputs_from_graph, outputs_from_graph, err)

		inputs = append(inputs, inputs_from_graph...)

		for _, input_name := range inputs {
			op := g.Operation(input_name)
			if op == nil {
				continue
			}

			input_shape := get_shape_by_reshape(op)
			if input_shape == nil {
				// we could not find Reshape right next to input placeholder, try to deduce its correct shape with heuristics
				channels_order := []int64{1, 4, 3}
				if strings.Index(input_name, "gray") != -1 {
					channels_order = []int64{1, 3, 4}
				} else if strings.Index(input_name, "rgba") != -1 {
					channels_order = []int64{4, 1, 3}
				} else if strings.Index(input_name, "rgb") != -1 {
					channels_order = []int64{3, 1, 4}
				}

				log.Infof("model: %s, input operation: %s, type: %v, num outputs: %d, channels order: %v", name, op.Name(), op.Type(), op.NumOutputs(), channels_order)

				output_shapes, err := get_output_shapes(op)
				if err != nil {
					log.Errorf("model: %s, input operation: %s, type: %v: could not get output shapes: %v", name, op.Name(), op.Type(), err)
					continue
				}

				for _, shape := range output_shapes {
					if len(shape) != 2 {
						continue
					}

					dims := shape[1]
					for _, num_channels := range channels_order {
						input_shape = check_dims(dims, num_channels)
						if input_shape != nil {
							break
						}
					}

					if input_shape != nil {
						break
					}
				}
			}

			log.Infof("model: %s, input operation: %s, type: %v, outputs: %d, input_shape: %v", name, input_name, op.Type(), op.NumOutputs(), input_shape)
			if input_shape != nil {
				input_shapes[input_name] = input_shape
			}
		}

		service := &ServiceModel{
			graph:   g,
			session: session,
			outputs: outputs_from_graph,
		}
		dev.AddService(name, service)
		serving_devices = append(serving_devices, dev)
		serving_slots += dev.config.MaxSlots
		bound_devices = append(bound_devices, fmt.Sprintf("%s:%d", dev_name, dev.config.MaxSlots))
	}

	if len(serving_devices) == 0 {
		return fmt.Errorf("could not bind service %s from %s to any device among %v, please check log file", name, model_file, devices)
	}

	ctx.serving_devices[name] = &ServingDevices{
		service_name: name,
		slots_available: serving_slots,
		devices: serving_devices,
		input_shapes: input_shapes,
		queue: list.New(),
	}

	ctx.slots_available = 0
	for _, dev := range ctx.devices {
		if len(dev.services) != 0 {
			// device is in use, count its max_slots into maximum number of parallel jobs supported by the session manager
			ctx.slots_available += dev.config.MaxSlots
		}
	}

	if ctx.slots_available > ctx.slots_available_from_config {
		log.Infof("session manager: number of slots as sum of all devices used in the services is %d, but config limits session manager to %d parallel slots",
			ctx.slots_available, ctx.slots_available_from_config)
		ctx.slots_available = ctx.slots_available_from_config
	}

	log.Infof("%s: added new service, number of available slots in the service: %d, in the session manager: %d, bound devices: %v", name, serving_slots, ctx.slots_available, bound_devices)

	return nil
}

func (ctx *SessionManager) Cleanup() {
	for _, dev := range ctx.devices {
		dev.Cleanup()
	}

	if ctx.session != nil {
		ctx.session.Close()
	}
}

type ExecutionSlot struct {
	ctx             *SessionManager
	dev             *Device
	serving_devices *ServingDevices
	num_requests    int

	service_model *ServiceModel
}

func (ctx *SessionManager) QueueLengths(service_name string) (int, int, int) {
	if len(service_name) == 0 {
		service_name = ctx.default_service_name
	}

	srv, ok := ctx.serving_devices[service_name]
	if !ok {
		return 0, 0, 0
	}

	ctx.lock.Lock()
	defer ctx.lock.Unlock()

	return srv.queue.Len(), srv.slots_available, ctx.slots_available
}

func (ctx *SessionManager) GetExecutionSlot(service_name string, num_requests int) (*ExecutionSlot, error) {
	if len(service_name) == 0 {
		service_name = ctx.default_service_name
	}

	srv, ok := ctx.serving_devices[service_name]
	if !ok {
		return nil, fmt.Errorf("there are no devices which can serve %s", service_name)
	}

	ctx.lock.Lock()
	defer ctx.lock.Unlock()


	if srv.slots_available <= 0 || ctx.slots_available <= 0 {
		cond := sync.NewCond(ctx.lock)
		elm := srv.queue.PushBack(cond)

		for srv.slots_available <= 0 || ctx.slots_available <= 0 {
			cond.Wait()

			if srv.slots_available <= 0 || ctx.slots_available <= 0 {
				log.Infof("%s: awakened, but there is no slot: srv_slots: %d, ctx_slots: %d", service_name, srv.slots_available, ctx.slots_available)
			}
		}

		srv.queue.Remove(elm)
	}

	var selected_dev *Device
	for _, dev := range srv.devices {
		if dev.slots_available > 0 {
			if selected_dev == nil || dev.config.Priority > selected_dev.config.Priority {
				selected_dev = dev
			}

			if selected_dev.config.Priority == dev.config.Priority && dev.slots_available > selected_dev.slots_available {
				selected_dev = dev
			}
		}
	}

	if selected_dev == nil {
		return nil, fmt.Errorf("could not find device for service %s among %d devices, which is impossible", service_name, len(srv.devices))
	}

	ctx.slots_available -= num_requests

	srv.slots_available -= num_requests
	selected_dev.slots_available -= num_requests

	service_model := selected_dev.services[service_name]

	slot := &ExecutionSlot{
		ctx:             ctx,
		dev:             selected_dev,
		serving_devices: srv,
		num_requests:    num_requests,
		service_model:   service_model,
	}

	return slot, nil
}

func (slot *ExecutionSlot) Cleanup() {
	if slot != nil {
		slot.ctx.lock.Lock()

		ctx := slot.ctx
		srv := slot.serving_devices
		dev := slot.dev

		ctx.slots_available += slot.num_requests
		srv.slots_available += slot.num_requests
		dev.slots_available += slot.num_requests

		if srv.slots_available > 0 && ctx.slots_available > 0 {
			if srv.queue.Len() > 0 {
				elm := srv.queue.Front()

				elm.Value.(*sync.Cond).Signal()
			}
		}

		slot.ctx.lock.Unlock()
	}
}

func (slot *ExecutionSlot) Session() *tf.Session {
	return slot.service_model.session
}

func (slot *ExecutionSlot) Graph() *tf.Graph {
	return slot.service_model.graph
}

func (slot *ExecutionSlot) ServiceName() string {
	return slot.serving_devices.service_name
}

func (slot *ExecutionSlot) InputShape(name string) []int64 {
	return slot.serving_devices.input_shapes[name]
}

func (slot *ExecutionSlot) NewSessionRun() *SessionRun {
	return &SessionRun{
		slot: slot,
		output_operations: make(map[string]*OutputOperation),
		outputs: make([]tf.Output, 0),
	}
}

func (run *SessionRun) OutputNames() []string {
	return run.slot.service_model.outputs
}

func (run *SessionRun) AddOutput(name string, convert ConversionFunction) error {
	op := run.slot.Graph().Operation(name)
	if op == nil {
		return fmt.Errorf("there is no output operation '%s'", name)
	}

	out_op := &OutputOperation {
		Name: name,
		Output: op.Output(0),
		Index: len(run.outputs),
		Convert: convert,
	}
	run.output_operations[name] = out_op
	run.outputs = append(run.outputs, out_op.Output)

	return nil
}

func (run *SessionRun) Output(name string) (interface{}, error) {
	op, ok := run.output_operations[name]
	if !ok {
		return nil, fmt.Errorf("there is no operation '%s'", name)
	}

	return op.Result, nil
}


func (run *SessionRun) check_and_convert(res []*tf.Tensor, op *OutputOperation) error {
	if op.Index >= len(res) {
		return fmt.Errorf("invalid number of results: %d, must be more than current operation's index %d", len(res), op.Index)
	}

	var err error
	op.Result, err = op.Convert(res[op.Index].Value())
	if err != nil {
		return fmt.Errorf("%s: check/convert error: %v", op.Name, err)
	}
	//log.Infof("%s: converted: %v", op.Name, op.Result)

	return nil
}

func (run *SessionRun) convert_results(res []*tf.Tensor) error {
	if len(res) != len(run.output_operations) {
		return fmt.Errorf("invalid number of results: %d, must be equal to number of operations: %d", len(res), len(run.output_operations))
	}

	for _, op := range run.output_operations {
		err := run.check_and_convert(res, op)
		if err != nil {
			return err
		}
	}

	return nil
}

func (run *SessionRun) Run(input_tensors map[string]*tf.Tensor) error {
	inputs := make(map[tf.Output]*tf.Tensor)

	for input_name, input_tensor := range input_tensors {
		op := run.slot.Graph().Operation(input_name)
		if op == nil {
			return fmt.Errorf("could not find input operation '%s'", input_name)
		}

		inputs[op.Output(0)] = input_tensor
	}

	res, err := run.slot.Session().Run(inputs, run.outputs, nil)
	if err != nil {
		return fmt.Errorf("sessioin.Run() failed: %v", err)
	}

	if len(res) != len(run.outputs) {
		return fmt.Errorf("session.Run() returned unexpected response, there are %d outputs, must be %d, outputs: %+v", len(res), len(run.outputs), res)
	}

	err = run.convert_results(res)
	if err != nil {
		return fmt.Errorf("conversion error between inputs and outputs: %v", err)
	}

	return nil
}
