# Biên dịch chaincode khi không truy cập được proxy.golang.org / google.golang.org / golang.org / gopkg.in

Ngày kiểm: 19/11/2026. Go 1.24.7. Mục đích: chứng minh `main.go` biên dịch và `go vet` sạch. Không dùng để triển khai.

## Kết quả

| Bước | Kết quả |
|---|---|
| `go build ./...` (bản sao có replace, dưới đây) | PASS |
| `go vet ./...` | PASS |
| `go test ./...` | `no test files` (chaincode chưa có test) |
| `go test ./merkle/` | PASS |

## Cách làm

Môi trường chỉ tải được từ `github.com`. Các module vanity (`google.golang.org`, `golang.org`, `gopkg.in`) bị chặn. Trong BẢN SAO làm việc (không commit), thêm các replace trỏ về GitHub:

```
replace golang.org/x/net => github.com/golang/net v0.28.0
replace golang.org/x/sys => github.com/golang/sys v0.24.0
replace golang.org/x/text => github.com/golang/text v0.17.0
replace google.golang.org/genproto => github.com/googleapis/go-genproto v0.0.0-20240814211410-ddb44dafa142
replace google.golang.org/genproto/googleapis/rpc => github.com/googleapis/go-genproto/googleapis/rpc v0.0.0-20240814211410-ddb44dafa142
replace google.golang.org/protobuf => github.com/protocolbuffers/protobuf-go v1.36.1
replace google.golang.org/grpc => github.com/grpc/grpc-go v1.67.0
replace gopkg.in/yaml.v3 => github.com/go-yaml/yaml/v3 v3.0.1
replace gopkg.in/check.v1 => github.com/go-check/check v0.0.0-20161208181325-20d25e280405
```

Môi trường: `GOPROXY=direct GOSUMDB=off GOFLAGS=-mod=mod`, không đặt `NO_PROXY` cho các host trên. Sau đó `go mod tidy -e` và `go build ./...`, `go vet ./...`.

## Hạn chế

- `go.mod` và `go.sum` của repo KHÔNG đổi. Build chính thức cần mạng tới proxy.golang.org (`go mod download`), chưa chạy được trong môi trường này.
- Phiên bản replace được chọn theo `go.mod` hiện tại. Nếu đổi phiên bản, kiểm lại.
- Chưa chạy chaincode trên Fabric thật.
