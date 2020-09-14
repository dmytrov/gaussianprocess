# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 16:16:56 2019

@author: johan
"""
import pickle
import pandas as pd 




favorite_color = pickle.load( open( "combined_errors.pkl", "rb" ) )
#df = pd.read_pickle("combined_elbo_stats.pkl")
#df = pd.DataFrame(df, columns='model')
#df.to_csv("koko.csv", encoding='ascii')
#print(df.to_string())
#df.type
df = pd.DataFrame(favorite_color)
df = df.stack()
df.columns = df.columns.droplevel(-2)

#df.pivot
#df = df.stack()
#df.pivot(index='foo', columns='bar', values='baz')
#print(df.to_string())
#print(set([a['model']['model'] for a in favorite_color]))
#df.to_csv('gogogo.csv')
