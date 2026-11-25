package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"
)

func newJob(ctx context.Context, query string, count int) *job {
	inputs := make([]pair, count)
	for i := range inputs {
		inputs[i] = pair{Query: query, Title: "example"}
	}
	return &job{ctx: ctx, pairs: inputs, added: time.Now(), done: make(chan result, 1)}
}

func TestBatchKeepsOrderAndSkipsCanceledWork(t *testing.T) {
	ctx := context.Background()
	first := newJob(ctx, "first", 2)
	second := newJob(ctx, "second", 1)
	deferred := newJob(ctx, "next batch", 2)
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	b := &batcher{ctx: ctx, queue: make(chan *job, 3), maxPairs: 4,
		wait: time.Second, workerTimeout: time.Second,
		score: func(_ context.Context, inputs []pair) (scoreResponse, error) {
			if len(inputs) != 3 {
				t.Fatalf("worker received %d pairs, want 3", len(inputs))
			}
			queries := []string{inputs[0].Query, inputs[1].Query, inputs[2].Query}
			if !reflect.DeepEqual(queries, []string{"first", "first", "second"}) {
				t.Fatalf("worker got %v", queries)
			}
			return scoreResponse{ModelVersion: "test", Scores: []float64{0.1, 0.2, 0.3}}, nil
		},
	}
	b.queue <- newJob(canceled, "skip", 1)
	b.queue <- second
	b.queue <- deferred
	jobs, next := b.collect(first)
	if next != deferred || len(jobs) != 2 {
		t.Fatalf("expected two jobs now and one deferred, got %v and %v", jobs, next)
	}
	b.dispatch(jobs)
	for _, want := range []struct {
		work   *job
		scores []float64
	}{
		{first, []float64{0.1, 0.2}}, {second, []float64{0.3}},
	} {
		select {
		case got := <-want.work.done:
			if got.err != nil || got.batchPairs != 3 || !reflect.DeepEqual(got.scores, want.scores) {
				t.Fatalf("unexpected result %+v", got)
			}
		default:
			t.Fatal("missing result")
		}
	}
}

func TestRankOrdersTiesByID(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	b := &batcher{ctx: ctx, queue: make(chan *job, 1), maxPairs: 8, workerTimeout: time.Second,
		score: func(_ context.Context, inputs []pair) (scoreResponse, error) {
			return scoreResponse{ModelVersion: "test", Scores: []float64{0.8, 0.1, 0.8}}, nil
		},
	}
	go b.run()
	body := `{"query":"q","candidates":[{"id":"z","title":"A"},{"id":"b","title":"B"},{"id":"a","title":"C"}]}`
	w := httptest.NewRecorder()
	rankHandler(b, time.Second).ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/rank", strings.NewReader(body)))
	var got rankResponse
	if w.Code != http.StatusOK || json.Unmarshal(w.Body.Bytes(), &got) != nil || len(got.Results) != 3 {
		t.Fatalf("unexpected response %d %s", w.Code, w.Body.String())
	}
	ids := []string{got.Results[0].ID, got.Results[1].ID, got.Results[2].ID}
	if got.ModelVersion != "test" || !reflect.DeepEqual(ids, []string{"a", "z", "b"}) {
		t.Fatalf("unexpected ranking %+v", got)
	}
}

func TestRankRejectsOverloadAndInvalidRequests(t *testing.T) {
	ctx := context.Background()
	b := &batcher{ctx: ctx, queue: make(chan *job, 1), maxPairs: 32}
	b.queue <- newJob(ctx, "queued", 1)
	const good = `{"query":"q","candidates":[{"id":"a","title":"a"}]}`
	for _, row := range []struct {
		body string
		code int
	}{
		{good, http.StatusTooManyRequests},
		{good + strings.Repeat(" ", maxBody), http.StatusRequestEntityTooLarge},
		{good + ` {}`, http.StatusBadRequest},
		{`{"query":"q","candidates":[{"id":"a","title":"a"},{"id":"a","title":"b"}]}`, http.StatusBadRequest},
	} {
		w := httptest.NewRecorder()
		rankHandler(b, time.Second).ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/rank", strings.NewReader(row.body)))
		if w.Code != row.code {
			t.Fatalf("got %d, want %d: %s", w.Code, row.code, w.Body.String())
		}
	}
}

func TestRankDeadlineExpiresQueuedWork(t *testing.T) {
	ctx := context.Background()
	b := &batcher{ctx: ctx, queue: make(chan *job, 1), maxPairs: 8}
	body := `{"query":"q","candidates":[{"id":"a","title":"a"}]}`
	w := httptest.NewRecorder()
	rankHandler(b, 5*time.Millisecond).ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/rank", strings.NewReader(body)))
	if w.Code != http.StatusGatewayTimeout {
		t.Fatalf("got %d, want 504", w.Code)
	}
	work := <-b.queue
	if work.ctx.Err() != context.DeadlineExceeded {
		t.Fatalf("queued work did not expire: %v", work.ctx.Err())
	}
	b.dispatch([]*job{work})
}

func TestWorkerScorerRejectsMissingScore(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"model_version":"test","scores":[null]}`))
	}))
	defer server.Close()
	_, err := workerScorer(server.URL)(context.Background(), []pair{{Query: "one", Title: "two"}})
	if err == nil {
		t.Fatal("accepted a missing worker score")
	}
}
