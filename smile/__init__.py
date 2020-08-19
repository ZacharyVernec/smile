# Mostly empty file, required for python to treat this folder as a package

__all__ = ["global_params", "population", "sampling", "regression"]
#don't include example_populations.py nor helper.py

# TODOs (file-nonspecific):

#TODO printing methods
#TODO use Enums (?)
#TODO return self in population methods for chaining
#TODO remove unnecessary kwargs defaults (eg Population() should return an empty population, not one with size 10000)
#TODO black lines when plotting 'both' with 'day'
#TODO Parameter_generator class for easy printing?
#TODO make sure population can't be re-generated after being sampled
#TODO raise error if generate() is called with generate_parameters=False but no parameters had been generated before either
#TODO don't return filtered Population nor filtered PopulationList if copy=False
#TODO improve plotting syntax and design with seaborn
#TODO replace all numpy data with pandas dataframe
#TODO create fig and axes by default if no axes given for plotting methods
#TODO documentation: https://realpython.com/documenting-python-code/
#TODO possibility of slicing Population with scorename as first element in subscript tuple
#TODO keep track of expected value of parameters for use when plotting regression results
#TODO shape doesn't *need* to be the variable name for random, as long as there is 1
#TODO change calls to ndarray.flatten() into calls to either ndarray.ravel() which returns a view, or ndarray.flat which is an iterator
#TODO replace tile(arr, (npersons, 1)) with broadcast_to(arr, (npersons, len(arr)))
#TODO have population keep track of personnumbers

#TODO classify todolist into multiple categories