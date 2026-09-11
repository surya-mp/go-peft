package peft

// Config is validated before an adapter changes a model.
type Config interface {
	Validate() error
}
