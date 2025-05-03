# Project Plan: Local Testing and Pytest Suite

## Plan

1. **Finish SKIP_DB patching and local test run**
   - Ensure all AWS/DB dependencies are bypassed or stubbed when SKIP_DB=true.
   - Make sure the local test script runs fully without AWS credentials or DB access.
   - Refactor code as needed to make this robust and maintainable.

2. **Build out a pytest suite**
   - Convert the local test script (and other relevant tests) into pytest tests.
   - Use mocking/patching for all AWS, DB, and external dependencies (OpenAI, etc).
   - Ensure tests are isolated, fast, and do not require any real cloud credentials.

## Rationale
- This approach ensures both robust local/manual development and high-quality, maintainable automated tests.
- SKIP_DB patching allows for CLI/manual/interactive runs without cloud dependencies.
- Pytest + mocking enables true unit/integration testing with full control and speed.

## Next Steps
1. Patch all DB/AWS calls to respect SKIP_DB and stub or skip as needed.
2. Validate local test script works fully offline.
3. Scaffold and expand pytest suite with proper mocking.

---

# Task Plan: Async Story Orchestration Refactor

## Goal
Refactor the story generation orchestration logic so that:
- Orchestration lives in FableFactory (not lambda_function.py)
- LLM calls (TTS, image generation, etc.) are async and parallelized for speed
- generate_story_package is the main orchestration entrypoint
- All Lambda handlers use the new orchestration
- Tests use the new Lambda flow
- Remove lambda_function.py when complete

## Steps

- [x] 1. Analyze existing orchestration logic in lambda_function.py (`_create_story_package`) (done)
- [ ] 2. Implement `generate_story_package` in storytelling/narrative_engine.py as an async orchestration method of FableFactory
    - [x] a. Generate narrative text (done)
    - [x] b. Identify scenes (done)
    - [x] c. Generate images for each scene (async) (done)
    - [x] d. Generate narration for each scene (async) (done)
    - [x] e. Gather results and return a complete story package (done)
- [ ] 3. Expose a synchronous wrapper for Lambda compatibility
- [x] 4. Update lambdas/generate_book.py to use the new orchestration entrypoint (done)
- [x] 5. Update tests/local_test_script.py to use the new Lambda handlers (not lambda_function.py) (done)
- [x] 6. Delete lambda_function.py (done)
- [ ] 7. Confirm everything works with a successful test run
