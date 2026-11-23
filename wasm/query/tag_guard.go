//go:build js && wasm && !wasmjs

// Package main is the Parquet query program. This file makes a build without the
// wasmjs tag fail with a clear message.
package main

// go-parquet only provides its browser HTTP range reader under the wasmjs build tag.
var _ int = "build with: GOOS=js GOARCH=wasm go build -tags wasmjs"
