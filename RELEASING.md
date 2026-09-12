# Releasing

1. Run the standard and relevant bridge tests.
2. Update `CHANGELOG.md` for user-visible changes.
3. Commit the release state and push `main`.
4. Create and push an annotated semantic-version tag.
5. Verify `go list -m github.com/surya-mp/go-peft@<tag>` from a clean module.
