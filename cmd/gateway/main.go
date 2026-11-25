package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"sort"
	"strings"
	"syscall"
	"time"
	"unicode/utf8"
)

const (
	maxBody        = 256 << 10
	maxCandidates  = 32
	workerMaxPairs = 64
)

type candidate struct {
	ID    string `json:"id"`
	Title string `json:"title"`
}

type rankRequest struct {
	Query      string      `json:"query"`
	Candidates []candidate `json:"candidates"`
}

type rankedCandidate struct {
	ID    string  `json:"id"`
	Score float64 `json:"score"`
}

type rankResponse struct {
	ModelVersion string            `json:"model_version"`
	Results      []rankedCandidate `json:"results"`
	BatchPairs   int               `json:"batch_pairs"`
	QueueMS      float64           `json:"queue_ms"`
	WorkerMS     float64           `json:"worker_ms"`
}

func decodeOne(body io.Reader, target any) error {
	decoder := json.NewDecoder(body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	if err := decoder.Decode(new(json.RawMessage)); err == nil {
		return errors.New("expected a single JSON object")
	} else if err != io.EOF {
		return err
	}
	return nil
}

func nonblank(text string, limit int) bool {
	return strings.TrimSpace(text) != "" && utf8.RuneCountInString(text) <= limit
}

func (r rankRequest) pairs(limit int) ([]pair, error) {
	if !nonblank(r.Query, 512) {
		return nil, errors.New("query must contain 1 to 512 characters")
	}
	if len(r.Candidates) == 0 || len(r.Candidates) > limit {
		return nil, fmt.Errorf("provide 1 to %d candidates", limit)
	}
	seen := make(map[string]bool, len(r.Candidates))
	inputs := make([]pair, 0, len(r.Candidates))
	for _, item := range r.Candidates {
		if !nonblank(item.ID, 128) || !nonblank(item.Title, 2048) {
			return nil, errors.New("each candidate needs an id and a title within the length limits")
		}
		if seen[item.ID] {
			return nil, errors.New("candidate ids must be unique")
		}
		seen[item.ID] = true
		inputs = append(inputs, pair{Query: r.Query, Title: item.Title})
	}
	return inputs, nil
}

func sendJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func problem(w http.ResponseWriter, status int, message string) {
	sendJSON(w, status, map[string]string{"error": message})
}

func millis(value time.Duration) float64 {
	return float64(value) / float64(time.Millisecond)
}

func rankHandler(b *batcher, timeout time.Duration) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var input rankRequest
		if err := decodeOne(http.MaxBytesReader(w, r.Body, maxBody), &input); err != nil {
			var tooLarge *http.MaxBytesError
			if errors.As(err, &tooLarge) {
				problem(w, http.StatusRequestEntityTooLarge, "request exceeds 256 KiB")
				return
			}
			problem(w, http.StatusBadRequest, "invalid ranking request JSON")
			return
		}
		inputs, err := input.pairs(min(maxCandidates, b.maxPairs))
		if err != nil {
			problem(w, http.StatusBadRequest, err.Error())
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), timeout)
		defer cancel()
		work := &job{ctx: ctx, pairs: inputs, added: time.Now(), done: make(chan result, 1)}
		select {
		case <-b.ctx.Done():
			problem(w, http.StatusServiceUnavailable, "gateway is shutting down")
			return
		case <-ctx.Done():
			return
		case b.queue <- work:
		default:
			problem(w, http.StatusTooManyRequests, "ranking queue is full")
			return
		}

		select {
		case <-b.ctx.Done():
			problem(w, http.StatusServiceUnavailable, "gateway is shutting down")
		case <-ctx.Done():
			if errors.Is(ctx.Err(), context.DeadlineExceeded) {
				problem(w, http.StatusGatewayTimeout, "ranking deadline exceeded")
			}
		case answer := <-work.done:
			if answer.err != nil {
				if errors.Is(answer.err, context.DeadlineExceeded) {
					problem(w, http.StatusGatewayTimeout, "scoring worker deadline exceeded")
					return
				}
				problem(w, http.StatusBadGateway, "scoring worker unavailable or returned invalid scores")
				return
			}
			results := make([]rankedCandidate, len(input.Candidates))
			for i, item := range input.Candidates {
				results[i] = rankedCandidate{ID: item.ID, Score: answer.scores[i]}
			}
			sort.Slice(results, func(i, j int) bool {
				if results[i].Score == results[j].Score {
					return results[i].ID < results[j].ID
				}
				return results[i].Score > results[j].Score
			})
			sendJSON(w, http.StatusOK, rankResponse{
				ModelVersion: answer.modelVersion, Results: results, BatchPairs: answer.batchPairs,
				QueueMS: millis(answer.queueTime), WorkerMS: millis(answer.workerTime),
			})
		}
	}
}

func workerScorer(endpoint string) scoreFunc {
	client := &http.Client{
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	return func(ctx context.Context, inputs []pair) (scoreResponse, error) {
		body, err := json.Marshal(struct {
			Pairs []pair `json:"pairs"`
		}{Pairs: inputs})
		if err != nil {
			return scoreResponse{}, err
		}
		request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			return scoreResponse{}, err
		}
		request.Header.Set("Content-Type", "application/json")
		response, err := client.Do(request)
		if err != nil {
			return scoreResponse{}, err
		}
		defer response.Body.Close()
		if response.StatusCode != http.StatusOK {
			return scoreResponse{}, fmt.Errorf("worker returned HTTP %d", response.StatusCode)
		}
		const maxReply = 32 << 10
		content, err := io.ReadAll(io.LimitReader(response.Body, maxReply+1))
		if err != nil {
			return scoreResponse{}, err
		}
		if len(content) > maxReply {
			return scoreResponse{}, errors.New("worker response exceeds size limit")
		}
		var wire struct {
			ModelVersion string     `json:"model_version"`
			Scores       []*float64 `json:"scores"`
		}
		if err := decodeOne(bytes.NewReader(content), &wire); err != nil {
			return scoreResponse{}, err
		}
		result := scoreResponse{ModelVersion: wire.ModelVersion, Scores: make([]float64, 0, len(wire.Scores))}
		for _, score := range wire.Scores {
			if score == nil {
				return scoreResponse{}, errors.New("worker returned a missing score")
			}
			result.Scores = append(result.Scores, *score)
		}
		return result, nil
	}
}

func main() {
	listen := flag.String("listen", "127.0.0.1:8082", "gateway listen address")
	endpoint := flag.String("worker", "http://127.0.0.1:8001/score", "worker scoring URL")
	maxPairs := flag.Int("batch-pairs", 32, "maximum pairs per worker request")
	depth := flag.Int("queue", 64, "maximum queued requests")
	wait := flag.Duration("batch-wait", 2*time.Millisecond, "wait for more requests; 0 disables batching")
	timeout := flag.Duration("request-timeout", 750*time.Millisecond, "per request ranking deadline")
	workerTimeout := flag.Duration("worker-timeout", 500*time.Millisecond, "worker batch deadline")
	flag.Parse()
	if *maxPairs < 1 || *maxPairs > workerMaxPairs || *depth < 1 ||
		*wait < 0 || *timeout <= 0 || *workerTimeout <= 0 || *wait >= *timeout {
		log.Fatal("invalid queue, batch, or deadline settings")
	}
	workerURL, err := url.Parse(*endpoint)
	if err != nil || (workerURL.Scheme != "http" && workerURL.Scheme != "https") || workerURL.Host == "" {
		log.Fatal("worker must be a valid HTTP or HTTPS URL")
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	b := &batcher{ctx: ctx, queue: make(chan *job, *depth), maxPairs: *maxPairs,
		wait: *wait, workerTimeout: *workerTimeout, score: workerScorer(*endpoint)}
	go b.run()
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		sendJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("POST /rank", rankHandler(b, *timeout))
	server := &http.Server{Addr: *listen, Handler: mux, ReadHeaderTimeout: 3 * time.Second,
		ReadTimeout: 5 * time.Second, WriteTimeout: 5*time.Second + *timeout, IdleTimeout: 30 * time.Second}
	stopped := make(chan struct{})
	go func() {
		<-ctx.Done()
		shutdown, stop := context.WithTimeout(context.Background(), 5*time.Second)
		defer stop()
		if err := server.Shutdown(shutdown); err != nil {
			log.Printf("shutdown: %v", err)
		}
		close(stopped)
	}()
	log.Printf("gateway listening on %s", *listen)
	if err := server.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
	<-stopped
}
