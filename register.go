package inference

import "go.k6.io/k6/js/modules"

const importPath = "k6/x/inference"

func init() {
	modules.Register(importPath, new(rootModule))
}
