@echo off
REM Parse Wikipedia corpus with streaming architecture (Windows batch file)
REM Memory-safe for 12.4M sentences

REM Default configuration (adjust as needed)
SET CORPUS=notebooks\data\extracted_articles.txt
SET PARSER=chart
SET MAX_SENTENCES=12377687
SET BATCH_SIZE=1000
SET NUM_WORKERS=4
SET CHECKPOINT_EVERY=50000
SET OUTPUT_DIR=data\wikipedia_parsed

echo ==========================================
echo Wikipedia Corpus Streaming Parser
echo ==========================================
echo.
echo Configuration:
echo   Corpus: %CORPUS%
echo   Parser: %PARSER%
echo   Max sentences: %MAX_SENTENCES%
echo   Workers: %NUM_WORKERS%
echo   Output: %OUTPUT_DIR%
echo.
echo Estimated time: ~48 hours (with 4 workers)
echo Memory usage: ~2-4GB (constant)
echo.
echo Press Ctrl+C to stop (progress will be saved)
echo.

python scripts\stream_parse_corpus.py ^
    --corpus %CORPUS% ^
    --parser-type %PARSER% ^
    --max-sentences %MAX_SENTENCES% ^
    --batch-size %BATCH_SIZE% ^
    --num-workers %NUM_WORKERS% ^
    --checkpoint-every %CHECKPOINT_EVERY% ^
    --output-dir %OUTPUT_DIR% ^
    --resume

echo.
echo ==========================================
echo Parsing complete!
echo ==========================================
echo.
echo Results:
echo   Batch files: %OUTPUT_DIR%\batches\
echo   Progress: %OUTPUT_DIR%\progress.json
echo   Statistics: %OUTPUT_DIR%\parse_stats.json
echo.
echo To consolidate to pickle (for A100 training):
echo   python scripts\stream_parse_corpus.py --consolidate-only --output-dir %OUTPUT_DIR%
