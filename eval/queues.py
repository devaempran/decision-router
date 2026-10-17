"""The routing task: CFPB complaint `Product` -> the option text shown to the model.
The option text is what the model reads, so each is a short description of the team that would handle it. Products that
the CFPB renamed over time (or that one team handles) share an option."""

# option description -> CFPB product names routed to it
TEAMS = {
    "credit reporting: errors on credit reports or credit scores, credit repair, identity theft on a credit file":
        ["Credit reporting", "Credit reporting or other personal consumer reports",
         "Credit reporting, credit repair services, or other personal consumer reports"],
    "debt collection: collectors contacting the consumer about a debt, validation of a debt, harassment":
        ["Debt collection"],
    "credit and prepaid cards: card charges, fees, disputes, interest, rewards, card fraud":
        ["Credit card", "Credit card or prepaid card", "Prepaid card"],
    "bank accounts: checking or savings accounts, deposits, holds, overdrafts, closed accounts":
        ["Checking or savings account", "Bank account or service"],
    "money transfers: wire or app transfers, remittances, virtual currency, money orders":
        ["Money transfer, virtual currency, or money service", "Money transfers"],
    "mortgages: home loans, servicing, escrow, foreclosure, refinancing":
        ["Mortgage"],
    "student loans: federal or private student loan servicing, repayment, forgiveness":
        ["Student loan"],
    "vehicle loans and leases: auto loans, repossession, lease-end charges":
        ["Vehicle loan or lease"],
}
OPTIONS_BY_PRODUCT = {product: option for option, products in TEAMS.items() for product in products}
QUESTION = "Which team should handle this consumer complaint?"
