# Bank Statement Analyzer - Combined Workflow

A comprehensive pipeline for analyzing bank statements that combines PDF parsing, data normalization, and summary generation into a single orchestrated workflow.

## Overview

This system combines four separate scripts into a unified pipeline:

1. **PDF Parsing** (`parse-gemini-hsbc.py`) - Uses Google Gemini AI to extract data from bank statement PDFs
2. **Redis Parallel Processing** (`run_parallel.py` + `worker_redis.py`) - Handles parallel processing of multiple PDFs using Redis Queue
3. **Normalization** (`normalize.py`) - Consolidates and standardizes parsed Excel files
4. **Summary Generation** (`summary_generator.py`) - Creates comprehensive analysis and scoring

## Features

- **Parallel Processing**: Uses Redis + RQ for concurrent PDF parsing
- **Comprehensive Analysis**: Generates detailed financial analysis including fraud detection
- **Standardized Output**: Consistent Excel format across different bank statement types
- **Error Handling**: Robust error handling and logging throughout the pipeline
- **Modular Design**: Can run individual phases or complete pipeline
- **Scalable**: Configurable number of workers for optimal performance

## Prerequisites

### System Requirements
- Python 3.8+
- Redis server running locally or accessible
- Sufficient memory for parallel processing (recommended: 8GB+)

### Installation

1. **Install Redis**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install redis-server
   
   # macOS
   brew install redis
   
   # Start Redis
   redis-server
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install -r requirements_combined.txt
   ```

3. **Set up Environment Variables**:
   Create a `.env` file with your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

### Basic Usage

```bash
# Analyze all PDFs in a folder
python bank_statement_analyzer.py /path/to/bank_statements --output-dir ./results

# Analyze a single PDF file
python bank_statement_analyzer.py statement.pdf --output-dir ./results

# Use more workers for faster processing
python bank_statement_analyzer.py /path/to/folder --workers 8 --output-dir ./results
```

### Advanced Options

```bash
# Custom timeout for large files (1 hour)
python bank_statement_analyzer.py /path/to/folder --job-timeout 3600 --output-dir ./results

# Run only specific phases
python bank_statement_analyzer.py /path/to/folder --phase parsing
python bank_statement_analyzer.py /path/to/folder --phase normalization
python bank_statement_analyzer.py /path/to/folder --phase summary
```

### Command Line Arguments

- `input`: Path to PDF file or folder containing PDF files (required)
- `--output-dir`: Output directory for all results (default: ./results)
- `--workers`: Number of parallel workers for PDF parsing (default: 4)
- `--job-timeout`: Timeout for individual parsing jobs in seconds (default: 2400)
- `--phase`: Run specific phase only: parsing, normalization, summary, or all (default: all)

## Output Structure

The system creates a structured output directory:

```
results/
├── parsed_excels/           # Individual parsed Excel files
│   ├── statement1.xlsx
│   ├── statement2.xlsx
│   └── ...
├── normalized/              # Consolidated data
│   └── consolidated_data.xlsx
├── final_results/           # Final analysis
│   └── bank_statement_analysis.xlsx
└── processing_report.json   # Processing summary
```

## Output Files

### Parsed Excel Files
Each PDF generates an Excel file with two sheets:
- **Metadata**: Account information, statement period, etc.
- **Transactions**: Transaction details with standardized columns

### Consolidated Data
- **Consolidated Transactions**: All transactions merged and normalized
- **Consolidated Metadata**: Aggregated account information

### Final Analysis
Comprehensive Excel file with multiple sheets:
- **Summary - Scorecard**: Key metrics and customer information
- **Month-wise Analysis**: Monthly financial trends
- **Scoring Details**: Detailed scoring metrics
- **EOD Balances**: End-of-day balance matrix
- **Recurring Credit/Debit**: Pattern analysis
- **Return Txn**: Returned transaction analysis
- **Fraud Check Sheet**: Fraud detection results
- **Xns**: Transaction details with payment modes

## Workflow Phases

### Phase 1: PDF Parsing
- Converts PDFs to images
- Uses Google Gemini AI for OCR and data extraction
- Handles fallback processing for complex layouts
- Generates individual Excel files for each PDF

### Phase 2: Normalization
- Consolidates multiple Excel files
- Standardizes column headers and data formats
- Sorts transactions chronologically
- Merges metadata across files

### Phase 3: Summary Generation
- Calculates comprehensive financial metrics
- Detects patterns and anomalies
- Generates fraud detection analysis
- Creates scoring and rating assessments

## Error Handling

The system includes comprehensive error handling:

- **Redis Connection**: Checks Redis availability before starting
- **Job Monitoring**: Monitors individual parsing jobs for failures
- **File Validation**: Validates input files and output generation
- **Logging**: Detailed logging to both console and file
- **Graceful Shutdown**: Proper cleanup of worker processes

## Performance Optimization

### Worker Configuration
- Default: 4 workers (adjust based on CPU cores and memory)
- Recommended: 1 worker per CPU core for optimal performance
- Memory usage: ~2-4GB per worker depending on PDF complexity

### Timeout Settings
- Default job timeout: 40 minutes
- Adjust based on PDF size and complexity
- Large files may need increased timeout values

### Redis Configuration
- Ensure Redis has sufficient memory for job queues
- Monitor Redis memory usage during processing
- Consider Redis persistence settings for long-running jobs

## Testing the Redis Workflow

Before running the full pipeline, you can test the Redis parallel processing components:

```bash
# Test Redis workflow components
python test_redis_workflow.py

# Test complete integration
python test_integration.py
```

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   ```
   Error: Redis connection failed
   Solution: Ensure Redis server is running (redis-server)
   ```

2. **Job Timeout**
   ```
   Error: Job timeout reached
   Solution: Increase --job-timeout parameter for large files
   ```

3. **Memory Issues**
   ```
   Error: Out of memory
   Solution: Reduce --workers count or increase system memory
   ```

4. **API Key Issues**
   ```
   Error: Google API key not found
   Solution: Set GOOGLE_API_KEY in .env file
   ```

### Debugging

Enable detailed logging by checking the `bank_analyzer.log` file:
```bash
tail -f bank_analyzer.log
```

## Integration with Streamlit

The combined workflow is designed to be easily integrated with a Streamlit web application:

```python
import streamlit as st
from bank_statement_analyzer import BankStatementAnalyzer

# Initialize analyzer
analyzer = BankStatementAnalyzer(output_dir="./temp_results")

# Upload files and run analysis
uploaded_files = st.file_uploader("Upload Bank Statement PDFs", type="pdf")
if uploaded_files:
    success = analyzer.run_complete_analysis(uploaded_files)
    if success:
        st.success("Analysis completed!")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `bank_analyzer.log`
3. Open an issue on the repository
4. Contact the development team

## Changelog

### Version 1.0.0
- Initial release of combined workflow
- Integration of all four original scripts
- Redis-based parallel processing
- Comprehensive error handling and logging
- Structured output organization
