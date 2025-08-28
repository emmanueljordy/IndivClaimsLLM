import random
import numpy as np
import typing as tp


def _pad(x, length: int, pad_value):
    """
    Prepend the pad value until the array reaches the specific length
    
    Args:
        x: List of token IDs to pad
        length: Target length
        pad_value: Token ID to use for padding (must be provided, no default)
    """
    return [pad_value] * (length - len(x)) + x


#
def _pad_tokens(tokens, pad_value):
    """
    Checks that all tensors in the list have the same length, pads them if necessary to the max length

    Args:
        tokens: List of Tensors
        pad_value: Token ID to use for padding

    Returns:
        List of Tensors, where each Tensor has the same length
    """
    max_length = len(max(tokens, key=len))
    tokens = [_pad(t, max_length, pad_value) for t in tokens]
    return tokens


class TabulaStart:
    """ Abstract super class Tabula Start

    Tabula Start creates tokens to start the generation process.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer, automatically downloaded from llm-checkpoint
        pad_token_id (int): ID of the padding token for this tokenizer
    """
    def __init__(self, tokenizer):
        """
        Initializes the super class.

        Args:
            tokenizer: Tokenizer from the HuggingFace library
        """
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def get_start_tokens(self, n_samples: int) -> tp.List[tp.List[int]]:
        """ Get Start Tokens

        Creates starting points for the generation process

        Args:
            n_samples: Number of start prompts to create

        Returns:
            List of n_sample lists with tokens
        """
        raise NotImplementedError("This has to be overwritten but the subclasses")


class CategoricalStart(TabulaStart):
    """ Categorical Starting Feature

    A categorical column with its categories is used as starting point.

    Attributes:
        start_col (str): Name of the categorical column
        population (list[str]): Possible values the column can take
        weights (list[float]): Probabilities for the individual categories

    """
    def __init__(self, tokenizer, start_col: str, start_col_dist: dict, 
                 all_columns: tp.List[str] = None, special_ids: dict = None):
        """ Initializes the Categorical Start

        Args:
            tokenizer: Tokenizer from the HuggingFace library
            start_col: Name of the categorical column
            start_col_dist: Distribution of the categorical column (dict of form {"Cat A": 0.8, "Cat B": 0.2})
            all_columns: List of all column names (for separator-aware logic)
            special_ids: Special token IDs (for separator-aware logic)
        """
        super().__init__(tokenizer)

        assert isinstance(start_col, str), ""
        assert isinstance(start_col_dist, dict), ""

        self.start_col = start_col
        self.population = list(start_col_dist.keys())
        self.weights = list(start_col_dist.values())
        self.all_columns = all_columns
        self.special_ids = special_ids

    def get_start_tokens(self, n_samples):
        start_words = random.choices(self.population, self.weights, k=n_samples)
        
        # Separator-aware: use <ROWSEP> for last column, <COLSEP> for others
        if self.all_columns and self.start_col in self.all_columns:
            is_last = (self.start_col == self.all_columns[-1])
            separator = "<ROWSEP>" if is_last else "<COLSEP>"
        else:
            # Fallback to old behavior if column info not available
            separator = "<COLSEP>"
            
        start_text = [f"{self.start_col} {s} {separator}" for s in start_words]
        start_tokens = _pad_tokens(self.tokenizer(start_text, add_special_tokens=False)["input_ids"], self.pad_token_id)
        return start_tokens


class ContinuousStart(TabulaStart):
    """ Continuous Starting Feature

    A continuous column with some noise is used as starting point.

    Attributes:
        start_col (str): Name of the continuous column
        start_col_dist (list[float]): The continuous column from the train data set
        noise (float): Size of noise that is added to each value
        decimal_places (int): Number of decimal places the continuous values have
    """
    def __init__(self, tokenizer, start_col: str, start_col_dist: tp.List[float],
                 noise: float = .01, decimal_places: int = 5, 
                 all_columns: tp.List[str] = None, special_ids: dict = None):
        """ Initializes the Continuous Start

        Args:
            tokenizer: Tokenizer from the HuggingFace library
            start_col: Name of the continuous column
            start_col_dist: The continuous column from the train data set
            noise: Size of noise that is added to each value
            decimal_places: Number of decimal places the continuous values have
            all_columns: List of all column names (for separator-aware logic)
            special_ids: Special token IDs (for separator-aware logic)
        """
        super().__init__(tokenizer)

        assert isinstance(start_col, str), ""
        assert isinstance(start_col_dist, list), ""

        self.start_col = start_col
        self.start_col_dist = start_col_dist
        self.noise = noise
        self.decimal_places = decimal_places
        self.all_columns = all_columns
        self.special_ids = special_ids

    def get_start_tokens(self, n_samples):
        start_words = random.choices(self.start_col_dist, k=n_samples)
        # start_words += np.random.normal(size=n_samples) * self.noise  # add noise to start words
        
        # Separator-aware: use <ROWSEP> for last column, <COLSEP> for others
        if self.all_columns and self.start_col in self.all_columns:
            is_last = (self.start_col == self.all_columns[-1])
            separator = "<ROWSEP>" if is_last else "<COLSEP>"
        else:
            # Fallback to old behavior if column info not available
            separator = "<COLSEP>"
            
        start_text = [f"{self.start_col} {format(s, f'.{self.decimal_places}f')} {separator}" for s in start_words]
        start_tokens = _pad_tokens(self.tokenizer(start_text, add_special_tokens=False)["input_ids"], self.pad_token_id)
        return start_tokens


class RandomStart(TabulaStart):
    """ Random Starting Features

    Random column names are used as start point. Can be used if no distribution of any column is known.

    Attributes:
        all_columns (List[str]): Names of all columns
    """
    def __init__(self, tokenizer, all_columns: tp.List[str], special_ids: dict = None):
        """ Initializes the Random Start

        Args:
            tokenizer: Tokenizer from the HuggingFace library
            all_columns: Names of all columns
            special_ids: Special token IDs (for separator-aware logic)
        """
        super().__init__(tokenizer)
        self.all_columns = all_columns
        self.special_ids = special_ids

    def get_start_tokens(self, n_samples):
        start_words = random.choices(self.all_columns, k=n_samples)
        
        # Separator-aware: determine correct separator for each randomly chosen column
        start_text = []
        for col in start_words:
            is_last = (col == self.all_columns[-1])
            separator = "<ROWSEP>" if is_last else "<COLSEP>"
            # Just "ColName " isn't a full cell; start with empty content and close it
            start_text.append(f"{col} {separator}")
            
        start_tokens = _pad_tokens(self.tokenizer(start_text, add_special_tokens=False)["input_ids"], self.pad_token_id)
        return start_tokens


class CanonicalStart(TabulaStart):
    """ Canonical Separator-Aware Start

    Forces generation to start from a specific column with the correct separator.
    This is critical when training without rotation - must start at first column.
    Uses the correct separator: <COLSEP> for non-final columns, <ROWSEP> for the last column.

    Attributes:
        start_col (str): Name of the starting column
        all_columns (List[str]): Names of all columns (needed for separator logic)  
        special_ids (dict): Special token IDs from tokenizer
        start_col_dist (dict|list): Distribution for generating realistic start values
    """
    def __init__(self, tokenizer, all_columns: tp.List[str], start_col: str, 
                 special_ids: dict, start_col_dist: tp.Union[dict, list] = None):
        """ Initializes the Canonical Start

        Args:
            tokenizer: Tokenizer from the HuggingFace library
            all_columns: Names of all columns in original order
            start_col: Name of the starting column
            special_ids: Dictionary with special token IDs (colsep_id, rowsep_id, pad_id)
            start_col_dist: Distribution of starting column (dict for categorical, list for continuous)
        """
        super().__init__(tokenizer)
        self.start_col = start_col
        self.all_columns = all_columns
        self.special_ids = special_ids
        self.start_col_dist = start_col_dist
        
        # Validate inputs
        assert start_col in all_columns, f"Start column {start_col} not found in {all_columns}"
        
    def get_start_tokens(self, n_samples):
        # Determine if this is the last column (affects separator choice)
        is_last_column = (self.start_col == self.all_columns[-1])
        separator = "<ROWSEP>" if is_last_column else "<COLSEP>"
        
        # Generate realistic start values
        if self.start_col_dist is None:
            # Fallback: just use column name with empty value
            start_values = [""] * n_samples
        elif isinstance(self.start_col_dist, dict):
            # Categorical: sample from distribution
            population = list(self.start_col_dist.keys())
            weights = list(self.start_col_dist.values())
            start_values = random.choices(population, weights, k=n_samples)
        elif isinstance(self.start_col_dist, list):
            # Continuous: sample from list of values  
            start_values = random.choices(self.start_col_dist, k=n_samples)
        else:
            # Fallback for unknown distribution type
            start_values = [""] * n_samples
            
        # Create start text with proper format: "ColumnName value <SEPARATOR>"
        start_text = []
        for value in start_values:
            if value == "":
                # Empty value case (just column name + space + separator) 
                text = f"{self.start_col} {separator}"
            else:
                # Normal case: column name + space + value + space + separator
                text = f"{self.start_col} {value} {separator}"
            start_text.append(text)
            
        # Tokenize and pad
        start_tokens = _pad_tokens(
            self.tokenizer(start_text, add_special_tokens=False)["input_ids"], 
            self.pad_token_id
        )
        
        return start_tokens
