package main

import (
	"context"
	"errors"
	"math"
	"time"
)

type pair struct {
	Query string `json:"query"`
	Title string `json:"title"`
}

type scoreResponse struct {
	ModelVersion string    `json:"model_version"`
	Scores       []float64 `json:"scores"`
}

type scoreFunc func(context.Context, []pair) (scoreResponse, error)

type job struct {
	ctx   context.Context
	pairs []pair
	added time.Time
	done  chan result
}

type result struct {
	scores       []float64
	modelVersion string
	batchPairs   int
	queueTime    time.Duration
	workerTime   time.Duration
	err          error
}

type batcher struct {
	ctx           context.Context
	queue         chan *job
	maxPairs      int
	wait          time.Duration
	workerTimeout time.Duration
	score         scoreFunc
}

func (b *batcher) run() {
	var next *job
	for b.ctx.Err() == nil {
		first := next
		if first == nil {
			select {
			case <-b.ctx.Done():
				return
			case first = <-b.queue:
			}
		}
		next = nil
		if first.ctx.Err() == nil {
			var jobs []*job
			jobs, next = b.collect(first)
			b.dispatch(jobs)
		}
	}
}

func (b *batcher) collect(first *job) ([]*job, *job) {
	jobs := []*job{first}
	count := len(first.pairs)
	if b.wait == 0 {
		return jobs, nil
	}
	timer := time.NewTimer(b.wait)
	defer timer.Stop()
	for count < b.maxPairs {
		select {
		case <-b.ctx.Done():
			return nil, nil
		case <-timer.C:
			return jobs, nil
		case next := <-b.queue:
			if next.ctx.Err() != nil {
				continue
			}
			if count+len(next.pairs) > b.maxPairs {
				return jobs, next
			}
			jobs = append(jobs, next)
			count += len(next.pairs)
		}
	}
	return jobs, nil
}

func validScores(response scoreResponse, count int) bool {
	if response.ModelVersion == "" || len(response.Scores) != count {
		return false
	}
	for _, score := range response.Scores {
		if math.IsNaN(score) || math.IsInf(score, 0) {
			return false
		}
	}
	return true
}

func (b *batcher) dispatch(jobs []*job) {
	var live []*job
	var inputs []pair
	for _, request := range jobs {
		if request.ctx.Err() == nil {
			live = append(live, request)
			inputs = append(inputs, request.pairs...)
		}
	}
	if len(inputs) == 0 || b.ctx.Err() != nil {
		return
	}
	ctx, cancel := context.WithTimeout(b.ctx, b.workerTimeout)
	started := time.Now()
	response, err := b.score(ctx, inputs)
	elapsed := time.Since(started)
	cancel()
	if err == nil && !validScores(response, len(inputs)) {
		err = errors.New("invalid scoring worker response")
	}
	offset := 0
	for _, request := range live {
		end := offset + len(request.pairs)
		answer := result{
			modelVersion: response.ModelVersion, batchPairs: len(inputs),
			queueTime: started.Sub(request.added), workerTime: elapsed, err: err,
		}
		if err == nil {
			answer.scores = response.Scores[offset:end]
		}
		request.done <- answer
		offset = end
	}
}
