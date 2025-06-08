## EnhancedRAG Documentation

### Configurable Context Directory

`DocumentLoader` now accepts an optional path when instantiated:

```python
loader = DocumentLoader(base_context_dir="/path/to/my/docs")
```

If no directory is provided, it defaults to `settings.CONTEXT_DIR`, which is
read from `.env`. This change allows the ingestion and generation pipelines to
operate on any valid directory without breaking existing behaviour.
