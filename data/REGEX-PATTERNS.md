# Some RegEx patterns shared in the SecretBench repo

|Pattern ID|Secret Type|Regular Expression|Source|
|--------|--------|--------|--------|
|65|AWS API Secret|\b([A-Za-z0-9+/]{40})[ \r\n'"\x60]|TruffleHog|
|71|Azure Client Secret|(?i)(%s).{0,20}([a-z0-9_\.\-~]{34})|TruffleHog|
|216| Dropbox API Key|\b(sl\.[A-Za-z0-9\-\_]{130,140})\b|TruffleHog|
|237|Facebook Access Token|EAACEdEose0cBA[0-9A-Za-z]+|Meli et al.|
|278|Generic Pattern|(?i)(?:pass\|token\|cred\|secret\|key)(?:.\|[\n\r]){0,40}(\b[\x21-\x7e]{16,64}\b)|TruffleHog|
|290|Github Token|\b((?:ghp\|gho\|ghu\|ghs\|ghr)_[a-zA-Z0-9]{36,255})\b|TruffleHog|
|605|Slack Token|(xoxb\|xoxp\|xapp\|xoxa\|xoxr)\-[0-9]{10,13}\-[a-zA-Z0-9\-]*|TruffleHog|
|640|Stripe API Key|[rs]k_live_[a-zA-Z0-9]{20,30}|TruffleHog|
|691|Twitter Access Token|(?i)(?:twitter)(?:.\|[\n\r]){0,40}\b[1-9][0-9]\+\-[0-9a-zA-Z]{40}\b|Meli et al.|
|747|Youtube/Google OAuth ID|[0-9]+-[0-9A-Za-z_]{32}\.apps\.googleusercontent\.com|Meli et al.|