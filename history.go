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

type HistoryStorage struct {
	OwnerId int32
	EnvId int32

	Entries *list.List

	MaxEntries int

	UpdateTime time.Time
}

func (h *History) NewHistoryStorage(owner_id, env_id int32) *HistoryStorage {
	return &HistoryStorage {
		OwnerId: owner_id,
		EnvId: env_id,
		Entries: list.New(),
		MaxEntries: h.MaxEntriesPerStorage,
	}
}

func (hs *HistoryStorage) AppendEntry(e *Entry) {
	if hs.Entries.Len() >= hs.MaxEntries {
		first := hs.Entries.Front()
		hs.Entries.Remove(first)
	}

	hs.Entries.PushBack(e)
	hs.UpdateTime = time.Now()
}

type History struct {
	sync.Mutex

	MaxEntriesPerStorage int

	NumEntries int

	Clients map[int32]*HistoryStorage

	PruneTimeout time.Duration
}

func NewHistory(max_entries_per_storage int, prune_timeout time.Duration) *History {
	return &History {
		MaxEntriesPerStorage: max_entries_per_storage,

		NumEntries: 0,

		Clients: make(map[int32]*HistoryStorage),

		PruneTimeout: prune_timeout,
	}
}

func (h *History) Append(n *halite_proto.HistoryEntry) {
	e := NewEntry(n)

	h.Lock()
	defer h.Unlock()

	idx := n.OwnerId * 1000 + n.EnvId

	hs, ok := h.Clients[idx]
	if !ok {
		hs = h.NewHistoryStorage(n.OwnerId, n.EnvId)
		h.Clients[idx] = hs
	}

	old_num_entris := hs.Entries.Len()
	hs.AppendEntry(e)
	h.NumEntries = h.NumEntries - old_num_entris + hs.Entries.Len()

	remove_clients := make([]int32, 0)
	for id, hs := range h.Clients {
		if time.Now().After(hs.UpdateTime.Add(h.PruneTimeout)) {
			log.Infof("removing client %d.%d (%d) because of lack of activity since %v, it has entries: %d",
				hs.OwnerId, hs.EnvId, id, hs.UpdateTime, hs.Entries.Len())
			remove_clients = append(remove_clients, id)
			h.NumEntries -= hs.Entries.Len()
		}
	}

	for _, id := range remove_clients {
		delete(h.Clients, id)
	}
}

// fixed trajectory len
func (h *History) Sample(trlen int, max_batch_size int) ([][]*Entry) {
	h.Lock()
	defer h.Unlock()

	total_len := 0
	episodes := make([]*list.List, 0, len(h.Clients))
	for _, hs := range h.Clients {
		if hs.Entries.Len() < trlen {
			continue
		}

		episodes = append(episodes, hs.Entries)
		total_len += hs.Entries.Len()
	}

	if total_len == 0 {
		return nil
	}

	ret := make([][]*Entry, 0, max_batch_size)
	for _, l := range episodes {
		trj := make([]*Entry, trlen, trlen)

		e := l.Back()
		for idx := trlen - 1; idx >= 0; idx -= 1 {
			trj[idx] = e.Value.(*Entry)

			e = e.Prev()
		}

		ret = append(ret, trj)

		if len(ret) > max_batch_size {
			break
		}
	}

	return ret
}
