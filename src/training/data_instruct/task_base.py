"""
Module containing iterable datasets used to finetune and test LSWTransformer models.
"""

from abc import ABC, abstractmethod
from enum import Enum

from datasets import DatasetDict, Dataset

class InstructionDatasetTask( Enum ):
    """ Enum of Instruction Dataset Types
    
    Properties:
        value (str): the string value representation of the task.
        description (str | None): a description describing the type of task.
    """
    
    SPAN_EXTRACTION = 'span_extraction', 'Span extraction task (Squad v1) where the question can always be answered with the given context.'
    SPAN_EXTRACTION_V2 = 'span_extraction_v2', 'Span extraction task (Squad v2) where the question may be answer using the given context or may be unanswerable.'
    MULTIPLE_CHOICE_OPEN = 'multiple_choice_open', 'Multiple choice question task with a provided context. The possible choices are enumerated in the question.'
    MULTIPLE_CHOICE_CLOSED = 'multiple_choice_closed', 'Multiple choice question without a provided context. The possible choices are enumerated in the question.'
    INSTRUCT_OPEN = 'instruct_open', 'Question answering task where the possible responses are given. Has explicit hard negatives.'
    INSTRUCT_CLOSED = 'instruct_closed', 'Question answering task where the response must be generated. Does not have explicit hard negatives.'
    
    def __new__( cls, *args, **kwargs ):
        obj = object.__new__( cls )
        obj._value_ = args[0]
        return obj
    
    def __init__( self, _: str, description: str ):
        self._description_ = description

    @property
    def description( self ) -> str:
        """ Short description of the task type """
        return self._description_



class BaseInstructDataset( ABC ):
    
    def __init__( self, cache_dir: str ):
        self._dataset = self.download( cache_dir )
    
    
    """ ========================================================================
        Utility functions
        ======================================================================== """
    
    @abstractmethod
    def download( self, cache_dir: str ) -> DatasetDict:
        """ Loads the dataset.
        
        Args:
            cache_dir (str): huggingface cache directory.
        """
    
    def __repr__( self ) -> str:
        
        group_name = f"'{self.group_name}'" if self.group_name is not None else 'None'
        task_name = f"'{self.task_name}'" if self.task_name is not None else 'None'
        
        return (
            f"{self.__class__.__name__}(\n"
            f"\ttask_type='{self.task_type.value}',\n"
            f"\ttask_description='{self.task_description}',\n"
            f"\tgroup_name={group_name},\n"
            f"\ttask_name={task_name},\n"
            f"\thas_training_docs={self.has_training_docs},\n"
            f"\thas_validation_docs={self.has_validation_docs},\n"
            f"\thas_test_docs={self.has_test_docs},\n"
            f"\thas_fewshot_docs={self.has_fewshot_docs},\n"
            f")"
        )
    
    """ ========================================================================
        Class properties
        ======================================================================== """
    
    @property
    @abstractmethod
    def task_type( self ) -> InstructionDatasetTask:
        """ Returns the task type """
    
    @property
    def task_description( self ) -> str:
        """ Returns the task description """
        return self.task_type.description
    
    @property
    @abstractmethod
    def group_name( self ) -> str | None:
        """ Returns the task group name or `None` if not a subtask """
    
    @property
    @abstractmethod
    def task_name( self ) -> str:
        """ Returns the specific task within the group """
    
    
    @property
    def has_training_docs( self ) -> bool:
        """ If the dataset has a training split """
        return self.get_training_docs() is not None
    
    @property
    def has_validation_docs( self ) -> bool:
        """ If the dataset has a validation split """
        return self.get_validation_docs() is not None
    
    @property
    def has_test_docs( self ) -> bool:
        """ If the dataset has a test split """
        return self.get_test_docs() is not None
    
    @property
    def has_fewshot_docs( self ) -> bool:
        """ If the dataset has a dedicated few-shot split """
        return self.get_fewshot_docs() is not None
    
    
    @abstractmethod
    def get_training_docs( self ) -> Dataset | None:
        """ Returns the training split if exists, else None """
    
    @abstractmethod
    def get_validation_docs( self ) -> Dataset | None:
        """ Returns the validation split if exists, else None """
    
    @abstractmethod
    def get_test_docs( self ) -> Dataset | None:
        """ Returns the test split if exists, else None """
    
    @abstractmethod
    def get_fewshot_docs( self ) -> Dataset | None:
        """ Returns the dedicated few-shot split if exists, else None """
    
    
    @property
    def dataset( self ) -> DatasetDict:
        """ Returns a reference to the underlying dataset """
        return self._dataset
    
    
    """ ========================================================================
        Message format functions
        ======================================================================== """
    
    def format_system_message( self, doc: dict ) -> dict[str, str | bool]:
        """ Creates a system message from a document.
        
        Args:
            doc (dict): the input document.
        
        Returns:
            dict: single line in the message list, ready to be formatted and tokenized.
        """
        _ = doc
        
        match self.task_type:
            case InstructionDatasetTask.SPAN_EXTRACTION:
                prompt = (
                    'Below is a question, paired with a background context. '
                    'Respond with text from the background that correctly answers the question.'
                )
        
            case InstructionDatasetTask.SPAN_EXTRACTION_V2:
                prompt = (
                    'Below is a question, paired with a background context. '
                    'Respond with text from the background that correctly answers the question. '
                    'If the questions cannot be answered using the background context, respond with "unanswerable".'
                )
            
            case InstructionDatasetTask.MULTIPLE_CHOICE_OPEN:
                prompt = (
                    'Below is a question, paired with a background context and multiple choices. '
                    'Respond with the choice that correctly answers the question.'
                )
                
            case InstructionDatasetTask.MULTIPLE_CHOICE_CLOSED:
                prompt = (
                    'Below is a question, paired with multiple choices. '
                    'Respond with the choice that correctly answers the question.'
                )
            
            case InstructionDatasetTask.INSTRUCT_OPEN:
                prompt = (
                    'Below is an instruction that describes a task. '
                    'Write a response that appropriately completes the request using the provided answer options.'
                )
            
            case InstructionDatasetTask.INSTRUCT_CLOSED:
                prompt = (
                    'Below is an instruction that describes a task. '
                    'Write a response that appropriately completes the request.'
                )
            
            case _:
                raise NotImplementedError( 'Subclasses should implement this system message' )
        
        return {
            'role': 'system',
            'content': prompt,
            'complete': True,
        }
    
    @abstractmethod
    def format_user_message( self, doc: dict ) -> dict:
        """ Creates a user input message from a document.
        
        Args:
            doc (dict): the input document.
        
        Returns:
            dict: single line in the message list, ready to be formatted and tokenized.       
        """
    
    @abstractmethod
    def format_target_messages( self, doc: dict ) -> list[dict]:
        """ Creates a list of candidate assistant output messages.
        
        Args:
            doc (dict): the input document.
        
        Returns:
            dict: list of message dicts.       
        """
    
    @abstractmethod
    def format_distractor_messages( self, doc: dict ) -> list[dict]:
        """ Creates a list of alternate or false assistant output messages.
        
        Args:
            doc (dict): the input document.
        
        Returns:
            dict: list of message dicts.       
        """
    
    @abstractmethod
    def format_unlabelled_messages( self, doc: dict ) -> list[dict]:
        """ Creates a list of all possible assistant output messages.
        
        Args:
            doc (dict): the input document.
        
        Returns:
            dict: list of message dicts.       
        """
    
    def format_generation_message( self, doc: dict ) -> dict:
        """ Creates an open ended output message from a document, ready for generation.
        
        Args:
            doc (dict): the input document.
        
        Returns:
            dict: single line in the message list, ready to be formatted and tokenized.       
        """
        _ = doc
        return {
            'role': 'assistant',
            'content': '',
            'complete': False,
        }
    
    
    """ ========================================================================
        Message chain functions
        ======================================================================== """
    
    def create_target_message_list( self, doc: dict ) -> list[list[dict]]:
        """ Creates a message list with the gold standard target. Use for SFT.
        
        Args:
            doc (dict): the input document.
        
        Returns:
            list[dict]: message list, ready to be formatted and tokenized.       
        """
        return [
            [
                self.format_system_message( doc ),
                self.format_user_message( doc ),
                target,
            ]
            for target in self.format_target_messages( doc )
        ]
    
    def create_distractor_message_list( self, doc: dict ) -> list[list[dict]]:
        """ Creates a list of message lists with the incorrect targets. Use for testing or contrastive training.
        
        Args:
            doc (dict): the input document.
        
        Returns:
            list[list[dict]]: list of message lists, ready to be formatted and tokenized.       
        """
        return [
            [
                self.format_system_message( doc ),
                self.format_user_message( doc ),
                target,
            ]
            for target in self.format_distractor_messages( doc )
        ]
    
    def create_unlabelled_message_list( self, doc: dict ) -> list[list[dict]]:
        """ Creates a list of message which may be true or false. Use for testing.
        
        Args:
            doc (dict): the input document.
        
        Returns:
            list[list[dict]]: list of message lists, ready to be formatted and tokenized.       
        """
        return [
            [
                self.format_system_message( doc ),
                self.format_user_message( doc ),
                target,
            ]
            for target in self.format_unlabelled_messages( doc )
        ]
    
    @abstractmethod
    def create_unlabelled_message_targets( self, doc: dict ) -> int | None:
        """ Returns the target index for binary and multi-class tasks.
        
        Args:
            doc (dict): the input document.
        
        Returns:
            int | None: The correct target index.      
        """
    
    def create_generation_messages( self, doc: dict ) -> list[dict]:
        """ Creates a message list with a (potentially) blank assistant message for answer generation.
        
        Args:
            doc (dict): the input document.
        
        Returns:
            list[dict]: message list, ready to be formatted and tokenized.       
        """
        return [
            self.format_system_message( doc ),
            self.format_user_message( doc ),
            self.format_generation_message( doc ),
        ]
    
    # TODO: get evaluation metric from HF hub?
    # TODO: test - create LLM queries?
    # TODO: test - process LLM query outputs?
    # TODO: train - produce SFT sequences?
    # TODO: actually load documents and retrieve
    # TODO: tokenization