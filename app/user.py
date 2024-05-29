from datetime import datetime
import pytz
import json
import logging
import os
import requests

CF_ACCESS_CLIENT_ID = os.environ.get('CF_ACCESS_CLIENT_ID')
CF_ACCESS_CLIENT_SECRET = os.environ.get('CF_ACCESS_CLIENT_SECRET')

def update_message_token_usage(user_id, message_id, message_type, llm_token_usage=0, embedding_token_usage=0) -> bool:
    return True
    
def get_user(user_id):
    user_info = {
        'user_type': 'premium',
        'premium_end_date': 1700000000,  # Example timestamp
        'llm_token_month_usage': 5000,
        'embedding_token_month_usage': 2000,
        'message_month_count': 100,
        'llm_token_today_usage': 100,
        'embedding_token_today_usage': 50,
        'message_today_count': 10,
        'payment_link': 'https://example.com/payment'
    }
    return user_info
    
def is_active_user(user_id):
    return True
        
def is_premium_user(user_id):
    return True
