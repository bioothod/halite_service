package main

import (
	"github.com/bioothod/halite/proto"
	"math/rand"
	"sync"
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
		Entries: make([]*Entry, 0, 100),
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
}

func NewHistoryStorage(owner_id, env_id int32, max_episodes int) *HistoryStorage {
	return &HistoryStorage {
		OwnerId: owner_id,
		EnvId: env_id,
		MaxEpisodes: max_episodes,
		Episodes: make([]*Episode, 0, max_episodes),
	}
}

func (hs *HistoryStorage) AppendEntry(e *Entry) {
	var ep *Episode
	if len(hs.Episodes) > 0 {
		ep = hs.Episodes[len(hs.Episodes) - 1]
	} else {
		ep = NewEpisode()
		hs.Episodes = append(hs.Episodes, ep)
	}

	if ep.Completed {
		ep = NewEpisode()
		hs.Episodes = append(hs.Episodes, ep)

		if len(hs.Episodes) > hs.MaxEpisodes {
			start := len(hs.Episodes) / 50 + 1
			if start > len(hs.Episodes) {
				start = len(hs.Episodes) / 2
			}

			hs.Episodes = hs.Episodes[start : len(hs.Episodes)]
		}
	}

	ep.Append(e)
}

type History struct {
	sync.Mutex

	MaxEpisodesPerStorage int
	MaxEpisodesTotal int
	NumEpisodes int

	Clients map[int32]*HistoryStorage
}

func NewHistory(max_episodes_per_storage, max_episodes_total int) *History {
	return &History {
		MaxEpisodesPerStorage: max_episodes_per_storage,
		MaxEpisodesTotal: max_episodes_total,
		NumEpisodes: 0,

		Clients: make(map[int32]*HistoryStorage),
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

	prev_episodes := len(hs.Episodes)
	hs.AppendEntry(e)
	diff := len(hs.Episodes) - prev_episodes

	h.NumEpisodes += diff
}

// fixed trajectory len
func (h *History) Sample(trlen int, max_batch_size int) ([]*Episode) {
	h.Lock()
	defer h.Unlock()

	num_episodes := 0
	for _, st := range h.Clients {
		num_episodes += len(st.Episodes)
	}

	episodes := make([]*Episode, 0, num_episodes)
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
	for _, ep := range episodes {
		start := rand.Intn(len(ep.Entries))
		end := len(ep.Entries)

		if end - start < trlen {
			start = end - trlen
		} else {
			end = start + trlen
		}

		copy_ep := &Episode {
			Completed: false,
			Entries: ep.Entries[start:end],
		}
		ret = append(ret, copy_ep)
	}

	return ret
}
