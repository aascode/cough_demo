import functools
import consul
import streamlit as st
import yaml

st.sidebar.subheader('Login Page')
username = st.sidebar.text_input('Username')
password = st.sidebar.text_input('Password', '******', type="password")



def get_config():
    with open('config.yaml', 'rb') as f:
        cfg = yaml.safe_load(f)
    return cfg


account = get_config()["account"]


def get_consult():
    return consul.Consul(host='localhost', port=80, scheme='http', verify=False)


class UI:
    def run(self): pass


class LogManagement(UI):
    name = "Breath Tracking Service"

    def __init__(self):
        c = get_consult()
        self.link = 'http://localhost:5601/app/discover#'

    def run(self):
        st.write('**Follow this link**')
        st.markdown(self.link, unsafe_allow_html=True)


def cache_on_button_press(label, **cache_kwargs):
    """Function decorator to memoize function executions.
    Parameters
    ----------
    label : str
        The label for the button to display prior to running the cached funnction.
    cache_kwargs : Dict[Any, Any]
        Additional parameters (such as show_spinner) to pass into the underlying @st.cache decorator.
    """
    internal_cache_kwargs = dict(cache_kwargs)
    internal_cache_kwargs['allow_output_mutation'] = True
    internal_cache_kwargs['show_spinner'] = False

    def function_decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            @st.cache(**internal_cache_kwargs)
            def get_cache_entry(func, args, kwargs):
                class ButtonCacheEntry:
                    def __init__(self):
                        self.evaluated = False
                        self.return_value = None

                    def evaluate(self):
                        self.evaluated = True
                        self.return_value = func(*args, **kwargs)

                return ButtonCacheEntry()

            cache_entry = get_cache_entry(func, args, kwargs)
            if not cache_entry.evaluated:
                if st.sidebar.button(label):
                    cache_entry.evaluate()
                else:
                    raise st.StopException
            return cache_entry.return_value

        return wrapped_func

    return function_decorator

def get_subclasses(cls):
    """returns all subclasses of argument, cls"""
    if issubclass(cls, type):
        subclasses = cls.__subclasses__(cls)
    else:
        subclasses = cls.__subclasses__()
    for subclass in subclasses:
        subclasses.extend(get_subclasses(subclass))
    return subclasses



@cache_on_button_press('Authenticate')
def authenticate(username, password):
    return username in account and account[username] == password


if authenticate(username, password):
    all_class = {x.name: x for x in get_subclasses(UI)}
    arr = list(all_class.keys())
    task = st.sidebar.selectbox('Choose service', arr)
    t = all_class[task]
    e = t()
    e.run()
else:
    st.sidebar.error('Authentication failed')