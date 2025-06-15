from remove_html_markup import remove_html_markup 
from Tracer import Tracer, ConditionalTracer, EventTracer

with EventTracer(events=['quote', 'tag']):
    remove_html_markup('<b title="bar">"foo"</b>')