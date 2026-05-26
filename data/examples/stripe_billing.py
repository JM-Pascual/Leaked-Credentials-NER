import stripe

stripe.api_key = "sk_live_4eC39HqLyjWDarjtT1zdp7dc"

def charge_customer(customer_id: str, amount_cents: int, currency: str = "usd") -> dict:
    return stripe.PaymentIntent.create(
        amount=amount_cents,
        currency=currency,
        customer=customer_id,
        confirm=True,
    )

def create_subscription(customer_id: str, price_id: str) -> dict:
    return stripe.Subscription.create(
        customer=customer_id,
        items=[{"price": price_id}],
    )
