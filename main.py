
# coding: utf-8

# In[1]:


from gspan_mining.config import parser
from gspan_mining.main import main


# In[2]:


#get_ipython().magic('pylab inline')


# In[3]:


args_str = '-s 2 -d True -l 5 -p True -w True graphdata/graph.data1'
FLAGS, _ = parser.parse_known_args(args=args_str.split())


# In[4]:


gs = main(FLAGS)


# ## plot graphs in database

# In[5]:


for g in gs.graphs.values():
    #g.plot()
    pass

