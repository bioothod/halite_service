package main

import (
	"github.com/bioothod/halite/proto"
	"math/rand"
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
	}

	return e
}

type Trajectory struct {
	entries []*Entry

	c_state []float32
	h_state []float32
}

func NewTrajectory(h *halite_proto.Trajectory) *Trajectory {
	tr := &Trajectory {
		entries: make([]*Entry, 0, len(h.Entries)),
		c_state: h.CState,
		h_state: h.HState,
	}

	for _, he := range h.Entries {
		e := NewEntry(he)

		tr.entries = append(tr.entries, e)
	}

	return tr
}

type history_storage struct {
	owner_id int32
	env_id int32

	history *History

	trajectories []*Trajectory

	update_time time.Time
}

func (h *History) Newhistory_storage(owner_id, env_id int32) *history_storage {
	return &history_storage {
		owner_id: owner_id,
		env_id: env_id,
		history: h,
		trajectories: make([]*Trajectory, 0),
		update_time: time.Now(),
	}
}

func (hs *history_storage) AppendTrajectory(tr *Trajectory) {
	hs.trajectories = append(hs.trajectories, tr)
	diff := len(hs.trajectories) - hs.history.max_trajectories_per_storage
	if diff > 0 {
		hs.trajectories = hs.trajectories[diff : len(hs.trajectories)]
	}

	hs.update_time = time.Now()
}

type History struct {
	sync.Mutex

	max_trajectories_per_storage int

	Clients map[int32]*history_storage

	prune_timeout time.Duration

	NumTrajectories int
}

func NewHistory(max_trajectories_per_storage int, prune_timeout time.Duration) *History {
	return &History {
		max_trajectories_per_storage: max_trajectories_per_storage,

		Clients: make(map[int32]*history_storage),

		prune_timeout: prune_timeout,

		NumTrajectories: 0,
	}
}

func generate_client_index(owner_id, env_id int32) int32 {
	return owner_id * 1000 + env_id
}

func (h *History) AddTrajectory(htr *halite_proto.Trajectory) {
	tr := NewTrajectory(htr)

	h.Lock()
	defer h.Unlock()

	idx := generate_client_index(htr.OwnerId, htr.EnvId)

	hs, ok := h.Clients[idx]
	if !ok {
		hs = h.Newhistory_storage(htr.OwnerId, htr.EnvId)
		h.Clients[idx] = hs
	}

	old_trnum := len(hs.trajectories)
	hs.AppendTrajectory(tr)
	h.NumTrajectories = h.NumTrajectories - old_trnum + len(hs.trajectories)

	remove_Clients := make([]int32, 0)
	for id, hs := range h.Clients {
		if time.Now().After(hs.update_time.Add(h.prune_timeout)) {
			log.Infof("removing client %d.%d (%d) because of lack of activity since %v, it has trajectories: %d",
				hs.owner_id, hs.env_id, id, hs.update_time, len(hs.trajectories))
			remove_Clients = append(remove_Clients, id)
		}
	}

	for _, id := range remove_Clients {
		delete(h.Clients, id)
	}
}

// fixed trajectory len
func (h *History) Sample(trlen int, max_batch_size int) ([][]*Entry, [][]float32, [][]float32) {
	h.Lock()
	defer h.Unlock()

	episodes := make([]*Trajectory, 0, len(h.Clients))

	for _, hs := range h.Clients {
		for _, tr := range hs.trajectories {
			if len(tr.entries) < trlen {
				continue
			}

			episodes = append(episodes, tr)
		}
	}

	if len(episodes) < max_batch_size {
		return nil, nil, nil
	}

	rand.Shuffle(len(episodes), func(i, j int) {
		episodes[i], episodes[j] = episodes[j], episodes[i]
	})

	ret_entries := make([][]*Entry, 0, max_batch_size)

	c_states := make([][]float32, 0, max_batch_size)
	h_states := make([][]float32, 0, max_batch_size)

	for _, tr := range episodes {
		ret_entries = append(ret_entries, tr.entries)
		c_states = append(c_states, tr.c_state)
		h_states = append(h_states, tr.h_state)

		if len(ret_entries) == max_batch_size {
			break
		}
	}

	return ret_entries, c_states, h_states
}
