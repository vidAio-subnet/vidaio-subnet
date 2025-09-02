# Discord Message Crawler

A Python script that monitors a Discord channel for messages from specific users and forwards them to your own Discord channel via webhook.

## Features

- Monitors a Discord channel for new messages from specified users
- Runs every minute (configurable)
- Sends new messages as rich embeds to your Discord webhook
- Tracks seen messages to avoid duplicates
- Handles mentions, attachments, and other message features
- Configurable via JSON file

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure the script:**
   
   Edit `config.json` with your settings:
   ```json
   {
       "channel_id": "YOUR_CHANNEL_ID_TO_MONITOR",
       "bot_token": "YOUR_BOT_TOKEN",
       "webhook_url": "YOUR_DISCORD_WEBHOOK_URL",
       "target_user_ids": [
           "USER_ID_1",
           "USER_ID_2",
           "USER_ID_3"
       ],
       "check_interval_seconds": 60
   }
   ```

3. **Get your Discord Bot Token:**
   - Go to [Discord Developer Portal](https://discord.com/developers/applications)
   - Create a new application or use existing one
   - Go to "Bot" section
   - Copy the token

4. **Get Channel ID:**
   - Enable Developer Mode in Discord (User Settings > Advanced > Developer Mode)
   - Right-click on the channel you want to monitor
   - Click "Copy ID"

5. **Get User IDs:**
   - Enable Developer Mode in Discord
   - Right-click on users you want to monitor
   - Click "Copy ID"

6. **Create Discord Webhook:**
   - Go to your target Discord channel
   - Channel Settings > Integrations > Webhooks
   - Create New Webhook
   - Copy the webhook URL

## Usage

### Option 1: Using config file (Recommended)
```bash
python discord_crawler_config.py
```

### Option 2: Direct configuration
Edit the variables in `discord_crawler.py` and run:
```bash
python discord_crawler.py
```

## Configuration Options

- `channel_id`: Discord channel ID to monitor
- `bot_token`: Your Discord bot token
- `webhook_url`: Discord webhook URL for notifications
- `target_user_ids`: List of user IDs to monitor
- `check_interval_seconds`: How often to check for new messages (default: 60)

## Message Format

The script sends messages as Discord embeds containing:
- Author information with avatar
- Message content
- Timestamp
- Channel reference
- Mentions (if any)
- Attachments (if any)
- Message ID for reference

## Example Output

When a target user posts a message, you'll see:
```
[2025-01-01 12:00:00] Checking for new messages...
Found new message from PaperMoney: got it
Successfully sent 1 message(s) to webhook
```

## Notes

- The script tracks message IDs to avoid sending duplicates
- It performs an initial fetch to populate the seen messages list
- Messages are limited to Discord's embed character limits
- The script handles API errors gracefully and continues running
- Use Ctrl+C to stop the crawler

## Troubleshooting

1. **"Failed to fetch messages"**: Check your bot token and channel ID
2. **"Failed to send webhook"**: Verify your webhook URL is correct
3. **No messages detected**: Ensure the user IDs are correct and users have posted recently
4. **Rate limiting**: Discord has rate limits; the script includes error handling for this

## Security

- Keep your bot token and webhook URL secure
- Don't commit these credentials to version control
- Consider using environment variables for production use