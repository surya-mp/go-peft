# go-peft
Parameter-Efficient Fine-Tuning for Go
Status: Phase 0–10 implemented; CUDA native validation is complete; benchmark publication remains

Current work:

1. [in progress] CUDA benchmark publication from the final merged NVIDIA worktree
2. [complete] Model-family profiles and injection dry-runs
3. [complete] Backend-neutral training loop utilities and CLI inspection
4. [complete] Base-model SafeTensors streaming and read-only GGUF v3 inspection; GGUF execution remains backend-specific
Initial implementation: LoRA
Long-term goal: LoRA, QLoRA, DoRA, and other PEFT methods
Primary language: Go
Design priority: Modular, backend-independent, interoperable, future-safe

## 1. Project Vision
go-peft is intended to become a modular Parameter-Efficient Fine-Tuning (PEFT) library for Go.
The project should provide reusable PEFT algorithms without coupling the core implementation to a specific machine-learning framework or tensor backend.

The initial implementation will focus exclusively on LoRA.

The architecture must, however, allow the project to grow naturally toward:

LoRA
QLoRA
DoRA
IA³
Prefix tuning
Prompt tuning
Other adapter-based PEFT methods
Multiple tensor/ML backends
Quantized model weights
Hugging Face-compatible adapters
Adapter composition
Adapter merging
Training and inference workflows
The primary architectural principle is:
Keep PEFT algorithms separate from tensor backends, model implementations, serialization, and quantization.
## 2. Goals
2.1 Primary Goals
The project should:
Provide a clean Go API for PEFT.
Implement LoRA correctly and efficiently.
Keep the core independent of a specific ML framework.
Support multiple backends.
Support adapter-only training.
Support adapter serialization.
Support SafeTensors where appropriate.
Support Hugging Face-compatible LoRA adapter formats.
Support architecture-neutral local base-model weight streaming.
Allow adapters to be merged and unmerged.
Provide strong unit and numerical tests.
Provide benchmarks.
Make QLoRA possible without redesigning the LoRA implementation.
Make future PEFT algorithms possible without breaking the public API.
## 3. Non-Goals for the Initial Release
The first release should NOT attempt to implement everything.
Do not initially implement:

Full model training framework
Dataset framework
Tokenizer framework
Optimizers
Complete quantization framework
Every PEFT method
Distributed training
CUDA runtime
Metal runtime

## 4. Core Architectural Principle
The project should be separated into four conceptual layers.
```text
                    go-peft
                       │
        ┌──────────────┼──────────────┐
        │              │              │
     Algorithms      Adapters      Formats
        │              │              │
      LoRA           Adapter       SafeTensors
      QLoRA          lifecycle     HF format
      DoRA           management       │
        │              │              │
        └──────────────┼──────────────┘
                       │
                    Backend
                       │
              ┌────────┼────────┐
              │        │        │
            GoMLX      MLX     Future
```


The important dependency direction is:
```text
PEFT algorithm
      ↓
backend abstraction
      ↓
concrete ML framework
```

NOT:
```text
GoMLX
  ↓
LoRA
  ↓
QLoRA
```

## 5. High-Level Architecture
The eventual project should resemble:
```text
go-peft/
│
├── lora/
│   ├── config.go
│   ├── adapter.go
│   ├── layer.go
│   ├── linear.go
│   ├── injection.go
│   ├── merge.go
│   ├── parameters.go
│   └── errors.go
│
├── peft/
│   ├── adapter.go
│   ├── config.go
│   ├── model.go
│   ├── parameter.go
│   └── registry.go
│
├── backend/
│   ├── backend.go
│   ├── tensor.go
│   ├── parameter.go
│   └── layer.go
│
├── format/
│   ├── safetensors/
│   └── huggingface/
│
├── qlora/
│   ├── config.go
│   ├── adapter.go
│   └── ...
│
├── quantization/
│   ├── scheme.go
│   ├── quantized.go
│   ├── nf4/
│   └── ...
│
├── backends/
│   ├── gomlx/
│   ├── mlx/
│   └── ...
│
├── examples/
│   ├── lora/
│   └── qlora/
│
├── internal/
│   └── testutil/
│
├── docs/
│   ├── architecture.md
│   ├── lora.md
│   ├── qlora.md
│   └── compatibility.md
│
├── go.mod
├── README.md
├── LICENSE
├── CONTRIBUTING.md
└── PLAN.md
```

However, this is the target architecture, not the initial implementation.

## 6. Initial Repository Structure
The first version should be intentionally small.
```text
go-peft/
│
├── go.mod
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── PLAN.md
│
├── peft/
│   ├── adapter.go
│   └── config.go
│
├── lora/
│   ├── config.go
│   ├── adapter.go
│   ├── linear.go
│   ├── injection.go
│   ├── merge.go
│   └── errors.go
│
├── backend/
│   ├── backend.go
│   └── tensor.go
│
├── examples/
│   └── lora/
│
└── internal/
    └── testutil/
```

Do not create empty QLoRA, DoRA, NF4, and backend packages simply because they are planned.
Create them when implementation begins.

## 7. Package Responsibilities
### 7.1 peft
This package defines concepts shared by all PEFT methods.
Responsibilities:

Adapter abstraction
PEFT configuration
Adapter lifecycle
Parameter metadata
Common errors
Adapter identification
It should NOT contain LoRA-specific mathematics.
Example conceptual API:

```code
type Adapter interface {
    Name() string
    Parameters() []Parameter
    Save(path string) error
}
```

The interface should remain minimal.
Avoid creating a large interface hierarchy prematurely.

## 8. LoRA Package
The lora package is the first major implementation.
Responsibilities:

LoRA configuration
LoRA adapter
LoRA-enabled linear layers
Adapter injection
Scaling
Initialization
Merge/unmerge
Trainable parameter identification
```formula
Core equation:
Y = XWᵀ + (α / r) X Aᵀ Bᵀ
```
Where:
```formula
W = frozen base weight
A = trainable low-rank matrix
B = trainable low-rank matrix
r = LoRA rank
α = LoRA alpha
```
The implementation should preserve this separation.

## 9. LoRA Configuration
Initial configuration should include:
```code
type Config struct {
    Rank           int
    Alpha          float32
    Dropout        float32
    TargetModules  []string
    Bias           BiasMode
}
```

Potential future fields:
```code
type Config struct {
    Rank                  int
    Alpha                 float32
    Dropout               float32
    TargetModules         []string
    Bias                  BiasMode

    Initialization        Initialization
    FanInFanOut            bool
    ModulesToSave         []string
    AdapterName            string
}
```
Do not add every possible option to v0.1.

Public configuration should grow only when there is a concrete requirement.

## 10. LoRA Layer
A LoRA-enabled linear layer conceptually contains:
```text
LoRALinear
│
├── Base Linear
│   └── W (frozen)
│
├── A
│   └── trainable
│
├── B
│   └── trainable
│
└── Scaling
    └── alpha / rank
```
The layer should not own the optimizer.
It should only expose parameters appropriately so that the host training framework can optimize them.

## 11. Parameter Ownership
One of the most important architectural requirements is distinguishing:
```text
Base parameters
```

from:
```text
Adapter parameters
```

Base parameters must be capable of being marked:
```text
Trainable = false
```

LoRA parameters must be:
```text
Trainable = true
```

The PEFT library should not assume how gradients or optimizer updates are implemented.
The host framework remains responsible for:

Forward pass execution
Autodiff
Gradient computation
Optimizer updates
Training loops
go-peft is responsible for identifying and exposing the correct trainable parameters.
## 12. Backend Abstraction
The backend abstraction exists to prevent the PEFT core from depending directly on GoMLX, MLX, or another tensor system.
The backend should expose only the operations necessary for PEFT.

Potential operations include:
```code
type Backend interface {
    MatMul(a, b Tensor) Tensor
    Add(a, b Tensor) Tensor
    Mul(a, b Tensor) Tensor
    Transpose(x Tensor) Tensor
    Shape(x Tensor) []int
}
```
This interface is only conceptual.
The actual interface should be determined after implementing the first backend.

Avoid abstracting operations that the project does not currently need.

## 13. Tensor Abstraction
Do not recreate an entire tensor library.
The backend should own tensor implementation.

Conceptually:
```code
type Tensor interface {
    Shape() []int
    DType() DType
}
```
The concrete backend can provide:
GoMLX Tensor
MLX Tensor
Future Tensor

The core should not inspect backend-specific tensor internals.
## 14. Backend Adapters
Concrete backend integrations should live outside the core algorithm.
Target architecture:
```text
backends/
├── gomlx/
│   └── ...
│
└── mlx/
    └── ...
```
For example:
adapter, err := gomlxlora.Inject(model, config)

or, preferably, an API where backend integration provides the model/layer bridge while the LoRA package owns the algorithm.
The exact API should be validated through implementation rather than prematurely finalized.

## 15. Model Integration
The project must support model traversal/injection without assuming a specific model architecture.
Conceptually:
```text
Model
│
├── layers.0
│   ├── q_proj
│   ├── k_proj
│   ├── v_proj
│   └── o_proj
│
├── layers.1
│   ├── q_proj
│   ├── k_proj
│   ├── v_proj
│   └── o_proj
│
└── ...
```
LoRA injection should be able to identify target modules such as:
q_proj
v_proj

without knowing whether the model is:
Llama
Qwen
Gemma
Mistral
another model family
a custom architecture
The model framework should provide module traversal.
go-peft should provide matching/injection logic.

## 16. Target Module Matching
Target module matching should support:
exact name
suffix matching
configured patterns

Example:
TargetModules: []string{
    "q_proj",
    "v_proj",
}

Potential future support:
layers.*.self_attn.q_proj

Do not introduce a complex pattern language unless necessary.
Simple deterministic matching is preferable initially.

## 17. Adapter Lifecycle
Adapters should have explicit lifecycle operations.
Conceptually:

Create
  ↓
Inject
  ↓
Train
  ↓
Save
  ↓
Load
  ↓
Use
  ↓
Merge

Possible states:
Unattached
Attached
Merged

Invalid transitions should produce clear errors.
For example:

merge twice
unmerge unmerged adapter
attach already attached adapter

should be handled safely.
## 18. Merge / Unmerge
LoRA merging should support:
W' = W + scaling × B × A

After merging:
base model
    ↓
W'

The LoRA computation can then be removed for inference.
The library should provide:

adapter.Merge()
adapter.Unmerge()

where supported.
Merging must be numerically tested.

The key invariant is:

output(before merge) ≈ output(after merge)

within an appropriate floating-point tolerance.
## 19. Multiple Adapters
The architecture should eventually support:
```text
Base Model
│
├── Adapter A
├── Adapter B
└── Adapter C
```
The initial release does not need sophisticated adapter composition.
However, avoid making the model architecture assume there can only ever be one adapter.

Future capabilities may include:

activate(adapter)
deactivate(adapter)
compose(adapterA, adapterB)
setWeight(adapter, 0.5)

## 20. Adapter Serialization
Serialization should be treated as a first-class component.
The project should aim for:
```text
adapter/
├── adapter_config.json
└── adapter_model.safetensors
```
The serialization layer should be separate from LoRA mathematics.
Conceptually:
```text
LoRA Adapter
     │
     ▼
Serialization interface
     │
     ├── SafeTensors
     ├── future formats
     └── memory/binary formats
```
This allows serialization formats to evolve independently.
## 21. Hugging Face Compatibility
Compatibility with the established PEFT ecosystem should be a major project goal.
The project should support importing/exporting compatible LoRA adapter configurations where practical.

Important metadata includes:

PEFT type
rank
alpha
target modules
adapter parameters
bias configuration
base model metadata where applicable
Compatibility should be tested against real adapter files.
Do not claim compatibility until files produced by the library have been tested against external tooling.

## 22. SafeTensors
SafeTensors support should be implemented independently from LoRA.
The project should eventually support:

Save adapter
Load adapter
Inspect tensors
Validate tensor shapes

Important requirements:
deterministic tensor naming
shape validation
dtype validation
missing tensor detection
unexpected tensor detection
safe failure on malformed files
## 23. Tensor Naming
Tensor names are part of interoperability.
Establish a stable convention early.

Example:

base_model.model.layers.0.self_attn.q_proj.lora_A.weight
base_model.model.layers.0.self_attn.q_proj.lora_B.weight

However, naming should follow established ecosystem conventions whenever possible.
Do not invent a custom naming scheme without a compatibility reason.

Create explicit tests for tensor naming.

## 24. Quantization Architecture
QLoRA requires the base model to remain quantized while LoRA parameters remain trainable in a higher precision.
Conceptually:
```text
Quantized Base
      │
      │ dequantize during computation
      ▼
   Forward
      │
      ├─────────────┐
      │             │
      ▼             ▼
  Base result    LoRA result
      │           FP16/BF16
      │             │
      └──────┬──────┘
             ▼
           Output
```
The quantization system must therefore be independent of LoRA.
Future structure:
```text
quantization/
├── scheme.go
├── quantized.go
├── nf4/
├── int4/
└── ...
```
## 25. Quantized Parameter Abstraction
Do not assume every parameter is a floating-point tensor.
A future parameter may represent:
```text
QuantizedParameter
│
├── quantized data
├── scale
├── metadata
└── quantization scheme
```
Potential schemes:
NF4
INT4
INT8

The LoRA layer should not need to understand how the base weight is quantized.
It should simply operate against a layer/backend capable of executing the required computation.

## 26. QLoRA
QLoRA should be implemented only after LoRA is stable.
Target architecture:
```text
qlora/
├── config.go
├── adapter.go
├── linear.go
└── training.go
```
QLoRA-specific features may eventually include:
4-bit quantized base weights
NF4
double quantization
paged optimizers
BF16/FP16 adapters
quantized linear layers
The first QLoRA milestone should not attempt all of these simultaneously.
Implement incrementally.

## 27. QLoRA Development Phases
QLoRA Phase 1
Support:
4-bit base weights
+
LoRA adapter

QLoRA Phase 2
Add:
NF4

QLoRA Phase 3
Add:
double quantization

QLoRA Phase 4
Investigate:
paged optimizer support

Optimizer functionality should remain outside the core PEFT package unless there is a strong architectural reason to include it.
## 28. DoRA
DoRA should eventually be implemented as a separate adapter algorithm.
Do not add DoRA-specific conditionals throughout LoRA.

Prefer:
```text
Adapter
├── LoRA
├── DoRA
└── future algorithms
```
This is one of the primary reasons the common PEFT abstractions should be small and generic.

## 29. General Adapter Model
Eventually the architecture should conceptually become:
```text
PEFT Model
│
├── Base Model
│
└── Adapter Registry
      │
      ├── LoRA
      ├── QLoRA
      ├── DoRA
      └── future
```
The adapter registry can provide:
Register(adapter)
Get(name)
Remove(name)
Activate(name)
Deactivate(name)

This does not need to exist in the first release.
## 30. Error Handling
Errors should be explicit and useful.
Examples:

ErrInvalidRank
ErrInvalidAlpha
ErrInvalidDropout
ErrUnsupportedLayer
ErrAdapterAlreadyAttached
ErrAdapterNotAttached
ErrAdapterAlreadyMerged
ErrAdapterNotMerged
ErrTensorShapeMismatch
ErrUnsupportedDType
ErrUnsupportedQuantization

Where appropriate, errors should wrap contextual information.
Example:

failed to inject LoRA into module "layers.3.self_attn.q_proj":
expected linear layer, got ...

## 31. Validation
Configuration should be validated before model mutation.
Examples:

rank > 0
alpha >= 0
dropout in [0, 1)
target modules not empty
supported dtype

Do not partially modify a model and then discover invalid configuration.
Preferred flow:

Validate
   ↓
Discover target modules
   ↓
Validate all targets
   ↓
Inject

This makes failures easier to reason about.
## 32. Initialization
The initial LoRA implementation should follow a well-defined initialization strategy.
Document:

A initialization
B initialization
scaling
dtype
device placement
The implementation should be deterministic when provided with a deterministic random source.
Tests should verify expected initialization properties.

## 33. Numerical Correctness
Numerical correctness is more important than API breadth.
Required tests should include:

Forward pass
Gradient flow
Scaling
Merge
Unmerge
Parameter freezing
Parameter shapes
Multiple target modules
Different ranks
Different alpha values

Numerical tests should compare:
LoRA implementation
vs
reference mathematical implementation

where possible.
## 34. Testing Strategy
Testing should be divided into:
Unit tests
Integration tests
Numerical tests
Compatibility tests
Benchmarks

Unit Tests
Test:
Config validation
Layer construction
Parameter shapes
Matching
State transitions
Numerical Tests
Test:
Forward computation
Merge equivalence
Unmerge equivalence
Scaling
Integration Tests
Test:
```text
real model
+
real backend
+
LoRA
+
training
```
Compatibility Tests
Test:
Go-generated adapter
→ external ecosystem

external adapter
→ Go

where supported.
## 35. Benchmarks
Benchmarks should be part of the project from early development.
Potential benchmarks:

BenchmarkLinearForward
BenchmarkLoRAForward
BenchmarkLoRAMerge
BenchmarkLoRAUnmerge
BenchmarkAdapterSave
BenchmarkAdapterLoad

Eventually:
BenchmarkLoRATraining
BenchmarkQLoRATraining
BenchmarkQuantizedForward

Record:
throughput
memory
parameter count
adapter size
merge time
load/save time
## 36. Parameter Efficiency Metrics
The library should provide or document how to calculate:
Base parameters
Trainable parameters
Trainable percentage
Adapter size

Example:
Base parameters:       7,000,000,000
Trainable parameters:     8,388,608
Trainable percentage:       0.12%

This is valuable both for users and for project benchmarks.
## 37. Documentation Requirements
Documentation should include:
```text
README.md
docs/
├── architecture.md
├── lora.md
├── serialization.md
├── compatibility.md
├── backends.md
└── qlora.md
```
The README should immediately show:
What the project is.
Why it exists.
Installation.
Minimal LoRA example.
Supported backends.
Compatibility.
Roadmap.
Avoid starting the README with implementation details.
## 38. Example Applications
Examples should demonstrate real workflows.
Initial example:

examples/lora/

Eventually:
```text
examples/
├── lora/
│   ├── minimal/
│   └── integration/
│
└── qlora/
    └── integration/
```
Examples should be kept small and executable.
## 39. API Stability
The project should use semantic versioning.
Before v1.0:

breaking changes are allowed

After v1.0:
public APIs require compatibility consideration

Keep internal implementation details private whenever possible.
Do not expose types merely because they are convenient internally.

## 40. Dependency Policy
Keep dependencies minimal.
The core package should not depend on:

A particular ML framework
A specific GPU runtime
A model repository
A tokenizer
A training framework
Backend packages may have backend-specific dependencies.
For example:
```text
go-peft
   │
   ├── core dependencies
   │
   └── optional backend integrations
```
This prevents dependency bloat.
## 41. Build Tags
Backend integrations may eventually use Go build tags where appropriate.
For example:

gomlx
mlx
cuda

However, do not introduce build-tag complexity until there is a concrete need.
Prefer simple modules when possible.

## 42. Backend Compatibility Matrix
Eventually maintain a compatibility table:
```table
Backend	LoRA	QLoRA	DoRA	Status
GoMLX	✓	NF4 ✓	Planned	Active
MLX	✓	Affine int4 ✓	Planned	Opt-in; requires MLX C runtime
Future backend	-	-	-	TBD
```
Only mark functionality as supported when tested.

## 43. Development Phases
Phase 0 — Repository Foundation
Tasks:
Create repository
Add Go module
Add license
Add README
Add PLAN.md
Establish package naming
Establish CI
Establish formatting/linting
Establish test workflow
Deliverable:
empty but healthy project

Phase 1 — Core Abstractions
Implement only the abstractions required by LoRA.
Tasks:

Adapter concept
Parameter metadata
Backend boundary
Tensor boundary
Common errors
Configuration validation
Deliverable:
small stable foundation

Do not attempt to create a complete generic ML abstraction.
Phase 2 — LoRA Mathematics
Implement:
LoRA config
LoRA parameters
A/B matrices
scaling
forward pass
initialization
Deliverable:
standalone LoRA layer

Phase 3 — Parameter Management
Implement:
frozen base parameters
trainable adapter parameters
parameter enumeration
parameter counting
Deliverable:
training-ready adapter

Phase 4 — Injection
Implement:
target matching
module discovery
layer replacement/wrapping
validation
safe injection
Deliverable:
```text
existing model
      ↓
LoRA-enabled model
```
Phase 5 — Merge
Implement:
merge
unmerge
state tracking
numerical tests
Deliverable:
```text
LoRA model
   ↓
merged inference model
```
Phase 6 — Serialization
Implement:
adapter config
deterministic tensor naming
SafeTensors
loading
saving
shape validation
Deliverable:
portable adapter

Phase 7 — First Real Backend
Choose one backend.
Recommended candidates:

GoMLX
MLX

Implement:
tensor bridge
model traversal
linear layer integration
training example
Deliverable:
real-world LoRA fine-tuning

Phase 8 — Second Backend
Implement a second backend.
This phase is important architecturally.

If the second backend requires significant changes to the core API, reconsider the abstraction before proceeding.

Deliverable:
```text
same PEFT API
+
multiple backends
```
Phase 9 — Ecosystem Compatibility
Implement and test:
Hugging Face adapter import
Hugging Face adapter export
SafeTensors compatibility
adapter metadata
compatibility documentation
Deliverable:
Go ↔ external PEFT ecosystem

Phase 10 — QLoRA
Only after LoRA is stable.
Implement:
```text
quantized parameter
      ↓
4-bit representation
      ↓
dequantization
      ↓
LoRA
```
Then incrementally:
```text
4-bit
↓
NF4
↓
double quantization
↓
paged optimizer investigation
```
Phase 11 — Additional PEFT Methods
Potential roadmap:
DoRA
IA³
Prefix tuning
Prompt tuning

Each method should be implemented as a separate algorithm rather than extending LoRA with conditionals.
## 44. Git Workflow
Use small, focused commits.
Recommended sequence:

feat: initialize project
feat: add PEFT adapter abstraction
feat: add LoRA configuration
feat: implement LoRA linear layer
test: add LoRA numerical tests
feat: add parameter management
feat: add module injection
feat: add merge and unmerge
feat: add adapter serialization
feat: add SafeTensors support
feat: add backend integration

Avoid massive commits such as:
"implement LoRA + QLoRA + serialization"

Small commits make the project easier to review and contribute to.
## 45. CI Requirements
CI should run:
go test ./...
go vet ./...
go fmt validation

Eventually:
golangci-lint
race tests
benchmarks
compatibility tests

Backend-specific CI should run where the required hardware/environment exists.
## 46. Release Strategy
Suggested releases:
v0.1.0
  Core LoRA

v0.2.0
  Injection + training integration

v0.3.0
  Serialization + SafeTensors

v0.4.0
  Backend integrations

v0.5.0
  External compatibility

v0.6.0+
  QLoRA development

v1.0.0
  Stable LoRA API

These versions are guidelines, not requirements.
Quality should determine releases rather than dates.

## 47. Community Strategy
The project should be designed for discoverability.
Important assets:

README
Examples
Benchmarks
Architecture documentation
Compatibility matrix
Roadmap
Good first issues
Contribution guide

Potential technical articles:
Implementing LoRA in Go
Building a Backend-Independent PEFT Library
QLoRA in Go: 4-bit Fine-Tuning
Running Parameter-Efficient Fine-Tuning on Apple Silicon with Go

The goal is not merely to publish code.
The goal is to build a recognizable Go PEFT ecosystem.

## 48. Open Source Quality Requirements
Before the first public release:
Clear README
License
API documentation
Examples
Unit tests
Numerical tests
CI
Error handling
Contribution guide
Changelog
Version tags
Avoid publishing a large amount of experimental code as the main API.
Keep experimental work under:
```text
internal/
```
or feature branches until stable.
## 49. Future-Proofing Rules
The following rules should guide development.
Rule 1
Do not make the core depend on a specific backend.
Rule 2
Do not recreate tensor frameworks.
Rule 3
Do not mix quantization logic with LoRA logic.
Rule 4
Do not mix optimizers with adapters.
Rule 5
Do not make adapters inseparable from models.
Rule 6
Do not assume all weights are floating-point.
Rule 7
Do not assume only one adapter can exist.
Rule 8
Do not invent serialization formats when ecosystem formats already exist.
Rule 9
Do not prematurely generalize.
Rule 10
Every abstraction should exist because a real implementation requires it.
## 50. Critical Design Boundary
The most important boundary in the entire project should be:
```text
             go-peft
                │
       ┌────────┴────────┐
       │                 │
   Algorithm          Backend
       │                 │
     LoRA              Tensor
     QLoRA             Execution
     DoRA              Model
       │                 │
       └────────┬────────┘
                │
              Host
            Framework
```

The algorithm should say:
"I need to multiply these tensors and add this adapter contribution."
The backend should decide:
"Here is how that tensor operation happens."
The host ML framework should decide:
"Here is how models, gradients, optimizers, datasets, and training are managed."
This separation is what allows the project to grow.
## 51. QLoRA Future-Safety Checklist
Before declaring LoRA architecture stable, verify:
 Base parameters can be frozen.
 Adapter parameters can be separately enumerated.
 Adapter parameters can have a different dtype from base weights.
 Base weights are not assumed to be directly mutable.
 Linear layers can be wrapped without assuming FP32/FP16 weights.
 Backend operations can work with quantized layers through an appropriate abstraction.
 Serialization can represent adapter tensors independently of base tensors.
 Adapter metadata does not assume unquantized models.
Model traversal does not assume a specific model family.
 Optimizer ownership remains outside the adapter implementation.
If these are true, the transition to QLoRA should be substantially easier.
## 52. First Milestone Definition
The first meaningful milestone is:
A mathematically correct LoRA linear layer with frozen base weights and trainable A/B matrices, backed by one real tensor runtime.
Acceptance criteria:
✓ Build succeeds
✓ Tests pass
✓ Forward computation is correct
✓ A/B shapes are correct
✓ Base weight is frozen
✓ A receives gradients
✓ B receives gradients
✓ Scaling is correct
✓ Multiple ranks work
✓ Merge is numerically equivalent
✓ Unmerge restores original behavior

Do not move to QLoRA until this is reliable.
## 53. Definition of Done for LoRA v1
LoRA should not be considered production-ready until:
[x] API documented
[x] Configuration validated
[x] LoRA linear implemented
[x] Parameter freezing implemented
[x] Injection implemented
[x] Merge/unmerge implemented
[x] Serialization implemented
[x] SafeTensors support implemented
[ ] External adapter compatibility tested independently with PEFT
[x] At least one backend supported
[x] At least one real model fine-tuned
[x] Numerical tests passing
[x] Integration tests passing
[x] CPU benchmarks published
[x] CI passing
[x] Example application available
[x] README complete

## 54. Definition of Done for QLoRA
Eventually:
[x] Quantized parameter abstraction
[x] 4-bit base weights
[x] NF4
[x] LoRA integration
[x] Correct gradient behavior
[x] Adapter remains higher precision
[ ] Memory benchmarks across native backends
[ ] Training benchmarks across native backends
[x] Double quantization
[ ] External compatibility tested independently with PEFT
[x] Real model example
[x] Documentation

## 55. Recommended Immediate Work
Do not start by implementing QLoRA.
Do not start by implementing SafeTensors.

Do not start by implementing every backend.

Start with:

1. Create repository
2. Create go.mod
3. Write README
4. Write architecture document
5. Define minimal PEFT Adapter abstraction
6. Define minimal backend boundary
7. Implement LoRA config
8. Implement LoRA linear mathematics
9. Write numerical tests
10. Integrate one backend

Then iterate.
## 56. Final Target Architecture
The eventual project should look conceptually like:
```text
                           go-peft
                              │
                  ┌───────────┴───────────┐
                  │                       │
               PEFT Core              Formats
                  │                       │
          ┌───────┼────────┐        ┌─────┴─────┐
          │       │        │        │           │
        LoRA    QLoRA    DoRA   SafeTensors   HF
          │       │        │
          │       │        │
          └───────┼────────┘
                  │
             Backend API
                  │
        ┌─────────┼─────────┐
        │         │         │
      GoMLX      MLX      Future
        │         │         │
        └─────────┼─────────┘
                  │
              ML Runtime
                  │
        ┌─────────┼─────────┐
        │         │         │
       CPU       GPU      Metal/CUDA
```

The project should remain modular enough that adding:
new PEFT method

does not require modifying:
existing LoRA code

and adding:
new backend

does not require rewriting:
PEFT algorithms

Likewise, adding:
new quantization method

should not require rewriting:
LoRA

## 57. Guiding Principle
The most important principle for the project is:
Build the smallest useful abstraction that does not prevent the next major capability.
For the first release:
```text
Go
 ↓
PEFT
 ↓
LoRA
 ↓
one backend
```
Then:
```text
Go
 ↓
PEFT
 ├── LoRA
 │
 └── QLoRA
      ↓
   quantization
```

And eventually:
```text
Go PEFT ecosystem
│
├── LoRA
├── QLoRA
├── DoRA
├── IA³
├── future PEFT methods
│
├── GoMLX
├── MLX
├── future backends
│
└── interoperable adapters
```
The first implementation should be small.
The architecture should be deliberate.

The public API should remain minimal.

The mathematics should be rigorously tested.

The formats should be interoperable.

And every major feature should be implemented as an independent module rather than becoming another conditional inside the original LoRA implementation.
