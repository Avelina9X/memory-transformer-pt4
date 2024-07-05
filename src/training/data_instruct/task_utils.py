""" Utilities and helper functions for finetuning tasks and datasets """

UNCENSOR_LIST = [
    'as an ai,',
    'as an ai ',
    'as a language model',
    'text-based AI assistant',
]

def phrase_filter( x: dict, content_key: str = 'content', messages_key: str = 'messages' ) -> bool:
    """ Filter function used for removing "censored" responses in SFT datasets.

    Args:
        x (dict): Dataset row
        content_key (str, optional): dict key for message content within the message list. Defaults to 'content'.
        messages_key (str, optional): dict key of the row for the messages list. Defaults to 'messages'.

    Returns:
        bool: Returns true if no "censored" phrases are found in the message thread.
    """
    return not any( any( phrase.lower() in line[content_key].lower() for phrase in UNCENSOR_LIST ) for line in x[messages_key] )