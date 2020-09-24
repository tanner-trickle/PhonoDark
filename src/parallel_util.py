# parallel helper functions

import numpy as np 

def generate_job_list(n_proc, total_job_list):
    """
    Will generate the job list to send to the processors.

    total_job_list - all of the jobs that need to be done
    """

    n_jobs = len(total_job_list)

    len_job = total_job_list.shape[1]

    if n_jobs > n_proc:

        n_extra_jobs = n_jobs // n_proc 

        print('  Number of jobs is greater than the the number of processors.')
        print()
        print('  Number of jobs per processor  = '+str(n_extra_jobs))
        print()

        job_list = np.zeros((n_proc, 1 + n_extra_jobs, len_job))

        count = 0

        for i in range(n_proc):

            for j in range(n_extra_jobs + 1):

                if count < n_jobs:
                    job_list[i, j, :] = total_job_list[count, :]
                else:
                    job_list[i, j, :] = -1*np.ones(len_job)

                count = count + 1

    elif n_jobs < n_proc:

        n_jobs_per_proc = 1

        job_list = np.zeros((n_proc, n_jobs_per_proc, len_job))

        print('\tNumber of jobs is less than the number of processors. '+
                        'Consider running with a smaller number of processors.')
        print('\tNumber of jobs = '+str(n_jobs))
        print()

        for i in range(n_proc):

            if i < n_jobs:

                job_list[i, 0, :] = total_job_list[i, :]

            else:

                job_list[i, 0, :] = -1*np.ones(len_job)

    else:

        print('\tNumber of jobs equal to the number of processors. Maximally parallized.')
        print()

        n_jobs_per_proc = 1
        job_list = np.zeros((n_proc, n_jobs_per_proc, len_job))

        job_list[:, 0, :] = total_job_list[:, :]

    return job_list 
