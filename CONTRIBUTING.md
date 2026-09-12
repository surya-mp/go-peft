# Contributing

Keep public APIs small and backend-neutral. Add tests beside the package they
cover, format with `gofmt`, and run `go vet ./...` plus `go test ./...`.

Run an optional bridge suite when changing that bridge:

```sh
go test -tags gomlx ./...
go test -tags mlx ./backends/mlx
go test -tags cuda ./backends/cuda
```
