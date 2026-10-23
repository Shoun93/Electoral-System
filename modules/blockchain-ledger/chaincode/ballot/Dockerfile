FROM golang:1.21-alpine AS build
WORKDIR /src
COPY . .
RUN go build -o /cc .

FROM alpine:3.19
COPY --from=build /cc /cc
CMD ["/cc"]
