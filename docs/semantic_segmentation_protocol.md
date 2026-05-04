# Semantic Segmentation Socket Protocol

This refactor uses option A: the semantics node sends prompts and class IDs once per
socket connection using a `configure` message, then sends only images in `segment`
messages.

Why this choice:

- It keeps the per-frame request small and stable.
- It matches SAM3 prompt caching well, so text embeddings are reused across images.
- It is still robust because the semantics node re-sends `configure` automatically
  after every reconnect.

Message flow:

1. `configure`
   - fields: `prompts`, `class_ids`
2. `configure_ack`
3. `segment`
   - fields: `image_bgr`
4. `segment_result`
   - fields: `class_map`

The segmentation server does pure inference only:

- it receives prompts and opaque class IDs
- it returns only `class_map uint8`
- it does not know colors, traversability, costs, or any other semantic attributes
