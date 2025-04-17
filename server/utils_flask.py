from flask_wtf import FlaskForm
from flask_wtf.form import _Auto
from wtforms import StringField, SubmitField, SelectField, RadioField, FieldList, FormField, TextAreaField, DecimalField
from wtforms.validators import DataRequired, NumberRange

class DatasetsForm(FlaskForm):
    dataset_name_text = StringField("Dataset Name")
    filepath_text = StringField("Dataset Filepath")
    total_datasets_text = StringField("Total Datasets")
    dataset_type_text = StringField("Dataset Type")

class updatedDatasetForm(FlaskForm):
    title = StringField("Available Datasets")
    field_list = FieldList(FormField(DatasetsForm), min_entries=0)

class TaskForm(FlaskForm):
    TaskId_text = StringField("Task_ID")
    Status_text = StringField("Status")
    Exec_Time_text = StringField("ExecutionTime")
    Description_text = TextAreaField("Description")

class updatedTaskForm(FlaskForm):
    title = StringField("Tasks")
    field_list = FieldList(FormField(TaskForm), min_entries=0)

class Task_Op_Model_Display(FlaskForm):
    rmse = StringField("rmse")
    mae = StringField("mae")
    mad = StringField("mad")
    exec_time = StringField("exec_time")

class updatedOpModelForm(FlaskForm):
    title = StringField("Operator Modelling")
    field_list = FieldList(FormField(Task_Op_Model_Display), min_entries=0)


class LoadDataForm(FlaskForm):
    Dataset_type = RadioField('Select Dataset Data Type', choices=[('1', 'Tabular Dataset'), ('2', 'Graphs ')], coerce=int)
    Dataset_DIR = StringField("Dataset Filepath", validators=[DataRequired()])
    Dataset_Name = StringField("Dataset Name", validators=[DataRequired()])
    submit = SubmitField("Submit")

    def __init__(self):
        super().__init__()
        self.dataset_type_dict = {'1': 'Tabular Dataset', '2':  'Graphs '}


class VectorisationForm(FlaskForm):
    select_dataset = SelectField('Select Dataset', choices=[], coerce=int)
    select_vec_size = SelectField('Select Vector Size', choices=[], coerce=int)
    save_into = SelectField('Save Vectors', choices=[], coerce=int)
    submit = SubmitField("Create Vectorization")
    
    def __init__(self, vector_sizes=None, avail_datasets=None, save_dbs=None):
        super().__init__()
        if vector_sizes:
            self.select_vec_size.choices = [(id, vect) for id, vect in vector_sizes.items()]
        if avail_datasets:
            self.select_dataset.choices = [(id, data_name) for id, data_name in avail_datasets.items()]
        if save_dbs:
            self.save_into.choices = [(id, ch) for id, ch in save_dbs.items()]

class DisplayVectorEmbeddingRepr(FlaskForm):
    select_ver = SelectField('Vector Embedding Representation', choices=[], coerce=int)
    submit = SubmitField("Display")
    
    def __init__(self, vec_emb_reprs=None):
        super().__init__()
        if vec_emb_reprs:
            self.select_ver.choices = [(id, ver) for id, ver in vec_emb_reprs.items()]


class SimilaritySearchForm(FlaskForm):
    sim_search_name = StringField("Name", validators=[DataRequired()], render_kw={'style': 'width: 50ch'})
    pred_data_dir_path = StringField("Prediction Datasets Directory Path", validators=[DataRequired()],  render_kw={'style': 'width: 50ch'})
    select_sim_search = RadioField('Select Similarity Search', choices=[], coerce=int)
    select_dataset = DecimalField('λ Dataset Selection Range (0, 1)', validators=[NumberRange(min=0.1, max=1.0, message='bla')])
    
    select_avail_vec_repr = SelectField('Select Available Vector Representation', choices=[], coerce=int)
    submit = SubmitField("Submit")

    def __init__(self, avail_sim_search=None, avail_vec_emb_reprs=None):
        super().__init__()
        self.select_sim_search.choices = [(id, sim_search) for id, sim_search in avail_sim_search.items()]
        self.select_avail_vec_repr.choices = [(id, avail_rep) for id, avail_rep in avail_vec_emb_reprs.items()]

class OperatorModellingForm(FlaskForm):
    operator_name = StringField("Name", validators=[DataRequired()])
    operator_dir = StringField("Output Directory Path", validators=[DataRequired()])
    select_avail_oper = SelectField('Select Operator', choices=[], coerce=int)
    select_avail_vec_repr = SelectField('Select Similarity Search', choices=[], coerce=int)

    submit = SubmitField("Operator Modelling")

    def __init__(self, vectors=None, operator_algorithms=None):
        super().__init__()
        self.select_avail_oper.choices = [(id, op_alg) for id, op_alg in operator_algorithms.items()]
        self.select_avail_vec_repr.choices = [(id, vec) for id, vec in vectors.items()]

class OperatorModellingDisplayForm(FlaskForm):
    avail_op_models = SelectField('Select Available Modelled Operators', choices=[], coerce=int)
    submit = SubmitField("Display")

    def __init__(self, avail_op_models_select=None):
        super().__init__()
        self.avail_op_models.choices = [(id, op_alg) for id, op_alg in avail_op_models_select.items()]

