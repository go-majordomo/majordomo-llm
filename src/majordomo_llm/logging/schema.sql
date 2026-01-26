-- Schema for LLM request logging
-- Works with both PostgreSQL and MySQL

CREATE TABLE IF NOT EXISTS llm_requests (
    request_id VARCHAR(36) PRIMARY KEY,
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    response_time FLOAT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cached_tokens INTEGER,
    input_cost DECIMAL(10, 8),
    output_cost DECIMAL(10, 8),
    total_cost DECIMAL(10, 8),
    s3_request_key VARCHAR(255),
    s3_response_key VARCHAR(255),
    status VARCHAR(20) NOT NULL,
    error_message TEXT
);

-- Optional indexes for common queries
CREATE INDEX idx_llm_requests_timestamp ON llm_requests(timestamp);
CREATE INDEX idx_llm_requests_provider ON llm_requests(provider);
CREATE INDEX idx_llm_requests_model ON llm_requests(model);
CREATE INDEX idx_llm_requests_status ON llm_requests(status);
