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

type Episode struct {
	Completed bool

	Entries []*Entry
}

func NewEpisode() *Episode {
	return &Episode {
		Completed: false,
		Entries: make([]*Entry, 0, 50),
	}
}

func (ep *Episode) Append(e *Entry) {
	ep.Entries = append(ep.Entries, e)
	if e.Done {
		ep.Completed = true
	}
}

type HistoryStorage struct {
	OwnerId int32
	EnvId int32

	MaxEpisodes int
	Episodes []*Episode

	NumEntries int

	UpdateTime time.Time
}

func NewHistoryStorage(owner_id, env_id int32, max_episodes int) *HistoryStorage {
	return &HistoryStorage {
		OwnerId: owner_id,
		EnvId: env_id,
		MaxEpisodes: max_episodes,
		NumEntries: 0,
		Episodes: make([]*Episode, 0, max_episodes),
	}
}

func (hs *HistoryStorage) AppendEntry(e *Entry) {
	if len(hs.Episodes) > hs.MaxEpisodes {
		start := len(hs.Episodes) / 10 + 1

		for idx := 0; idx < start; idx += 1 {
			hs.NumEntries -= len(hs.Episodes[idx].Entries)
		}

		hs.Episodes = hs.Episodes[start : len(hs.Episodes)]
	}

	var ep *Episode
	if len(hs.Episodes) > 0 {
		ep = hs.Episodes[len(hs.Episodes) - 1]
	} else {
		ep = NewEpisode()
		hs.Episodes = append(hs.Episodes, ep)
	}

	if ep.Completed && len(ep.Entries) > 100 {
		ep = NewEpisode()
		hs.Episodes = append(hs.Episodes, ep)
	} else {
		ep.Completed = false
	}

	ep.Append(e)
	hs.NumEntries += 1
	hs.UpdateTime = time.Now()
}

type History struct {
	sync.Mutex

	MaxEpisodesPerStorage int
	MaxEpisodesTotal int

	NumEntries int
	NumEpisodes int

	Clients map[int32]*HistoryStorage

	PruneTimeout time.Duration
}

func NewHistory(max_episodes_per_storage, max_episodes_total int, prune_timeout time.Duration) *History {
	return &History {
		MaxEpisodesPerStorage: max_episodes_per_storage,
		MaxEpisodesTotal: max_episodes_total,

		NumEntries: 0,
		NumEpisodes: 0,

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
		hs = NewHistoryStorage(n.OwnerId, n.EnvId, h.MaxEpisodesPerStorage)
		h.Clients[idx] = hs
	}

	old_num_entris := hs.NumEntries
	old_num_episodes := len(hs.Episodes)
	hs.AppendEntry(e)
	h.NumEntries = h.NumEntries - old_num_entris + hs.NumEntries
	h.NumEpisodes = h.NumEpisodes - old_num_episodes + len(hs.Episodes)

	remove_clients := make([]int32, 0)
	for id, hs := range h.Clients {
		if time.Now().After(hs.UpdateTime.Add(h.PruneTimeout)) {
			log.Infof("removing client %d.%d (%d) because of lack of activity since %v, it has entries: %d, episodes: %d",
				hs.OwnerId, hs.EnvId, id, hs.UpdateTime, hs.NumEntries, len(hs.Episodes))
			remove_clients = append(remove_clients, id)
			h.NumEpisodes -= len(hs.Episodes)
			h.NumEntries -= hs.NumEntries
		}
	}

	for _, id := range remove_clients {
		delete(h.Clients, id)
	}
}

// fixed trajectory len
func (h *History) Sample(trlen int, max_batch_size int) ([]*Episode) {
	h.Lock()
	defer h.Unlock()

	episodes := make([]*Episode, 0, h.NumEpisodes)
	for _, st := range h.Clients {
		for _, ep := range st.Episodes {
			if len(ep.Entries) > trlen {
				episodes = append(episodes, ep)
			}
		}
	}

	rand.Shuffle(len(episodes), func(i, j int) {
		episodes[i], episodes[j] = episodes[j], episodes[i]
	})

	if max_batch_size > len(episodes) {
		max_batch_size = len(episodes)
	}

	if max_batch_size == 0 {
		return nil
	}

	ret := make([]*Episode, 0, max_batch_size)
	for _, ep := range episodes[0 : max_batch_size] {
		end := len(ep.Entries)
		start := rand.Intn(len(ep.Entries))

		if false {
			start = end - trlen
		} else {
			if end - start < trlen {
				start = end - trlen
			} else {
				end = start + trlen
			}
		}

		copy_ep := &Episode {
			Completed: false,
			Entries: ep.Entries[start:end],
		}
		ret = append(ret, copy_ep)
	}

	return ret
}
