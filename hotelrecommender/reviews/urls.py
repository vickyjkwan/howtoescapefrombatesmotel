from django.conf.urls import url
from . import views

#app_name = 'hotelrecommender'

urlpatterns = [
    # ex: /
    url(r'^$', views.review_list, name='review_list'),
    # ex: /review/5/
    url(r'^review/(?P<review_id>[0-9]+)/$', views.review_detail, name='review_detail'),
    # ex: /hotel/
    url(r'^hotel$', views.hotel_list, name='hotel_list'),
    # ex: /hotel/5/
    url(r'^hotel/(?P<hotel_id>[0-9]+)/$', views.hotel_detail, name='hotel_detail'),
    url(r'^hotel/(?P<hotel_id>[0-9]+)/add_review/$', views.add_review, name='add_review'),
]
