import pandas as pd
import tables

def write_to_h5(filename, df):
    print 'Writing dataframe into {0}...'.format(filename)
    h5file = tables.open_file(filename, 'w', driver='H5FD_CORE')
    record = df.astype(float).to_records(index=False)
    filters= tables.Filters(complevel=5)        ## setting compression level
    h5file.create_table(h5file.root, 'DATA', filters=filters, obj=record)
    h5file.close()
    
def load_from_h5(filename, table='/DATA'):
    print "Loading dataframe from {0}...".format(filename)
    with tables.open_file(filename) as fp:
        df = pd.DataFrame(fp.get_node(table).read())
    return df

def _test():
    import os
    from IPython.display import display
    df_init = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })
    print "dataframe for testing: "
    display(df_init)
    
    write_to_h5('test.h5', df_init)
    df_final = load_from_h5('test.h5')
    
    print "dataframe after h5io: "
    display(df_final)
    
    print "The two dataframe is equal? \n", df_init == df_final
    
    print "removing the test file..."
    os.system('rm test.h5')

if __name__ == '__main__':
    _test()