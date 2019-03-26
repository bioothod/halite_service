package main

import (
	"container/list"
	"github.com/bioothod/halite/proto"
	"rndgit.msk/goservice/log"
	"sync"
	"time"
)

type State struct {
	State []byte
	Params []byte
}

type Entry struct {
	OldState	*State
	NewState	*State
	Done		bool
	Reward		float32
	Action		int32
	Step		int32
	Logits		[]float32
	TrainStep	int32
}

func NewEntry(n *halite_proto.HistoryEntry) *Entry {
	e := &Entry {
		OldState: &State {
			State: n.State.State,
			Params: n.State.Params,
		},
		NewState: &State {
			State: n.NewState.State,
			Params: n.NewState.Params,
		},

		Done: n.Done,
		Reward: n.Reward,
		Action: n.Action,
		Logits: n.Logits,

		Step: n.Step,
		TrainStep: n.TrainStep,
	}

	return e
}

type HistoryStorage struct {
	OwnerId int32
	EnvId int32

	Entries []*list.List

	NumEntries int
	MaxEntries int

	UpdateTime time.Time

	c_state []float32
	h_state []float32
}

func (h *History) NewHistoryStorage(owner_id, env_id int32) *HistoryStorage {
	return &HistoryStorage {
		OwnerId: owner_id,
		EnvId: env_id,
		Entries: make([]*list.List, 0),
		MaxEntries: h.MaxEntriesPerStorage,

		c_state: make([]float32, h.CStateSize),
		h_state: make([]float32, h.HStateSize),
	}
}

func (hs *HistoryStorage) Cleanup(num int) {
	if num <= 0 {
		return
	}

	new_start_index := 0
	for _, trj := range hs.Entries {
		first := trj.Front()
		if first != nil {
			for {
				next := first.Next()
				trj.Remove(first)
				num -= 1
				hs.NumEntries -= 1

				if next == nil || num < 0 {
					break
				}

				first = next
			}
		}

		if trj.Len() == 0 {
			new_start_index += 1
		}

		if num < 0 {
			break
		}
	}

	if new_start_index != 0 {
		hs.Entries = hs.Entries[new_start_index : len(hs.Entries)]
	}
}

func get_train_step(l *list.List) int32 {
	var train_step int32 = -1

	last := l.Back()
	if last != nil {
		last_ent := last.Value.(*Entry)
		train_step = last_ent.TrainStep
	}

	return train_step
}

func (hs *HistoryStorage) Append(e *Entry) {
	var l *list.List

	if len(hs.Entries) == 0 {
		l = list.New()
		hs.Entries = append(hs.Entries, l)
	} else {
		l = hs.Entries[len(hs.Entries) - 1]
		train_step := get_train_step(l)

		if e.TrainStep != train_step {
			l = list.New()
			hs.Entries = append(hs.Entries, l)
		}
	}

	l.PushBack(e)
	hs.NumEntries += 1
}

func (hs *HistoryStorage) AppendEntry(e *Entry) {
	hs.Append(e)
	hs.Cleanup(hs.NumEntries - hs.MaxEntries)

	hs.UpdateTime = time.Now()
}

type History struct {
	sync.Mutex

	MaxEntriesPerStorage int

	NumEntries int

	Clients map[int32]*HistoryStorage

	PruneTimeout time.Duration

	CStateSize int
	HStateSize int
}

func NewHistory(max_entries_per_storage int, prune_timeout time.Duration, c_state_size, h_state_size int) *History {
	return &History {
		MaxEntriesPerStorage: max_entries_per_storage,

		NumEntries: 0,

		Clients: make(map[int32]*HistoryStorage),

		PruneTimeout: prune_timeout,

		CStateSize: c_state_size,
		HStateSize: h_state_size,
	}
}

func generate_client_index(owner_id, env_id int32) int32 {
	return owner_id * 1000 + env_id
}

func (h *History) Append(n *halite_proto.HistoryEntry) {
	e := NewEntry(n)

	h.Lock()
	defer h.Unlock()

	idx := generate_client_index(n.OwnerId, n.EnvId)

	hs, ok := h.Clients[idx]
	if !ok {
		hs = h.NewHistoryStorage(n.OwnerId, n.EnvId)
		h.Clients[idx] = hs
	}

	old_num_entris := hs.NumEntries
	hs.AppendEntry(e)
	h.NumEntries = h.NumEntries - old_num_entris + hs.NumEntries

	remove_clients := make([]int32, 0)
	for id, hs := range h.Clients {
		if time.Now().After(hs.UpdateTime.Add(h.PruneTimeout)) {
			log.Infof("removing client %d.%d (%d) because of lack of activity since %v, it has entries: %d",
				hs.OwnerId, hs.EnvId, id, hs.UpdateTime, hs.NumEntries)
			remove_clients = append(remove_clients, id)
			h.NumEntries -= hs.NumEntries
		}
	}

	for _, id := range remove_clients {
		delete(h.Clients, id)
	}
}

// fixed trajectory len
func (h *History) Sample(trlen int, max_batch_size int, train_step int32) ([][]*Entry, []int32, [][]float32, [][]float32) {
	h.Lock()
	defer h.Unlock()

	episodes := make([]*list.List, 0, len(h.Clients))
	active_clients := make([]int32, 0, len(h.Clients))

	for _, hs := range h.Clients {
		for _, trj_list := range hs.Entries {
			trj_train_step := get_train_step(trj_list)

			if trj_train_step < train_step || trj_list.Len() < trlen {
				continue
			}

			episodes = append(episodes, trj_list)
			active_clients = append(active_clients, generate_client_index(hs.OwnerId, hs.EnvId))
		}
	}

	if len(episodes) < max_batch_size {
		return nil, nil, nil, nil
	}

	ret_entries := make([][]*Entry, 0, max_batch_size)

	client_indexes := make([]int32, 0, max_batch_size)

	c_states := make([][]float32, 0, max_batch_size)
	h_states := make([][]float32, 0, max_batch_size)

	for episode_idx, l := range episodes {
		client_id := active_clients[episode_idx]

		e := l.Front()
		trj := make([]*Entry, 0, trlen)

		for idx := 0; idx < trlen; idx += 1 {
			trj = append(trj, e.Value.(*Entry))

			e = e.Next()
		}

		ret_entries = append(ret_entries, trj)
		hs := h.Clients[client_id]
		c_states = append(c_states, hs.c_state)
		h_states = append(h_states, hs.h_state)

		client_indexes = append(client_indexes, client_id)

		if len(ret_entries) == max_batch_size {
			break
		}
	}

	return ret_entries, client_indexes, c_states, h_states
}

func (h *History) UpdateStates(states [][][]float32, indexes []int32) {
	h.Lock()
	defer h.Unlock()

	for idx, client_id := range indexes {
		hs, ok := h.Clients[client_id]
		if !ok {
			continue
		}

		hs.c_state = states[0][idx]
		hs.h_state = states[1][idx]
	}
}
