from behave import *

@given('model_name as string')
def step_impl(context):
    context.model_name = context.text