# VidaIO Subnet Dashboard API

A FastAPI-based REST API server that provides real-time information about the VidaIO subnet, including miner statistics and network data.

## Base URL
```
http://localhost:8000
```

## API Endpoints

### 1. Health Check
**Endpoint:** `GET /health`

**Description:** Verify API server is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1703123456.789,
  "service": "VidaIO Subnet Dashboard API"
}
```

**Use Case:** Frontend health monitoring, connection testing.

---

### 2. Miner Counts
**Endpoint:** `GET /miner_counts`

**Description:** Get current count of miners by processing task type with automatic retry logic.

**Response (Success):**
```json
{
  "status": "success",
  "compression_count": 45,
  "upscaling_count": 23,
  "total_count": 68,
  "attempt": 1,
  "timestamp": 1703123456.789
}
```

**Response (Error):**
```json
{
  "status": "error",
  "message": "Failed after 3 attempts: Database file not found: video_subnet_validator.db",
  "compression_count": 0,
  "upscaling_count": 0,
  "total_count": 0,
  "attempt": 3,
  "timestamp": 1703123456.789
}
```

**Retry Logic:**
- Maximum 3 attempts
- 1 second delay between retries
- Automatic retry on database connection failures

**Use Case:** Dashboard statistics, miner distribution charts, real-time monitoring.

---

### 3. Miner Information
**Endpoint:** `GET /miner_info`

**Description:** Get detailed information about all miners below the minimum stake threshold.

**Response (Success):**
```json
{
  "status": "success",
  "data": {
    "miners": [
      {
        "miner_uid": 123,
        "miner_hotkey": "5F...",
        "trust": 0.85,
        "incentive": 0.92,
        "emission": 0.78,
        "daily_reward": 15.6
      }
    ]
  },
  "timestamp": 1703123456.789
}
```

**Response (Error):**
```json
{
  "status": "error",
  "message": "Failed to retrieve miner information: Network connection failed",
  "data": {
    "miners": []
  },
  "timestamp": 1703123456.789
}
```

**Use Case:** Miner management interface, performance analytics, reward calculations.

---

## Data Models

### Miner Counts Response
| Field | Type | Description |
|-------|------|-------------|
| `status` | string | "success" or "error" |
| `compression_count` | integer | Number of compression miners |
| `upscaling_count` | integer | Number of upscaling miners |
| `total_count` | integer | Total number of miners |
| `attempt` | integer | Retry attempt number (1-3) |
| `timestamp` | float | Unix timestamp of response |
| `message` | string | Error message (only on error) |

### Miner Info Response
| Field | Type | Description |
|-------|------|-------------|
| `status` | string | "success" or "error" |
| `data.miners[]` | array | Array of miner objects |
| `timestamp` | float | Unix timestamp of response |
| `message` | string | Error message (only on error) |

### Miner Object
| Field | Type | Description |
|-------|------|-------------|
| `miner_uid` | integer | Unique miner identifier |
| `miner_hotkey` | string | Miner's public key |
| `trust` | float | Trust score (0.0 - 1.0) |
| `incentive` | float | Incentive score (0.0 - 1.0) |
| `emission` | float | Emission score (0.0 - 1.0) |
| `daily_reward` | float | Calculated daily reward |

## Error Handling

### HTTP Status Codes
- `200 OK`: Successful response
- `500 Internal Server Error`: Server-side errors

### Error Response Format
All error responses include:
- `status: "error"`
- `message`: Human-readable error description
- `timestamp`: When the error occurred
- Fallback data (empty arrays/zero values)

### Common Error Scenarios
1. **Database Connection Failed**: Retry automatically (up to 3 times)
2. **Table Not Found**: Check database configuration
3. **Network Issues**: Retry with exponential backoff
4. **Invalid Data**: Validation errors with specific messages
