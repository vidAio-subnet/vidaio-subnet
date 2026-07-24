# Compression warmup fixture

`compression_warmup_input.mp4` is a five-second, 320×180 synthetic test pattern
generated with FFmpeg's `testsrc2` source. It contains no third-party footage or
audio and may be redistributed with this repository.

Regenerate it with:

```bash
ffmpeg -f lavfi -i "testsrc2=size=320x180:rate=24:duration=5" \
  -an -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  competitions/fixtures/compression_warmup_input.mp4
```

The competition preflight validates that the fixture exists, contains a video
stream, is no longer than 5.5 seconds, and is no larger than 720p. The unscored
warmup requires `/compress` to preserve dimensions and timing while emitting a
smaller AV1 MP4.
