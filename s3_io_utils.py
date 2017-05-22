import base64
import hashlib
from urlparse import urlparse

import boto3
import gzip
import logging
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import os
import Queue
import random
import tensorflow as tf
import threading

from cStringIO import StringIO


FLAGS = tf.app.flags.FLAGS
LOG_FORMAT = '%(asctime)-15s %(message)s'
S3_BUCKET = 'pinlogs'

logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
logger = logging.getLogger('tf_io_utils')
tls = threading.local()


def s3_client():
    if not hasattr(tls, 's3'):
        # use threadlocal storage for the s3 client, since boto s3 client is not threadsafe
        tls.s3 = boto3.client('s3')
    return tls.s3


def log_info(msg, source=''):
    logger.info('%-15s %s' % (source, msg))


def log_error(msg, source=''):
    logger.info('%-15s %s' % (source, msg))


def parse_s3_url(s3_url):
    """Convenience function for parsing S3 URL's into separate bucket and path, needed by boto API

    :param s3_url: full S3 URL
    :return: bucket_name, object_key (S3 Path)
    """
    parsed_s3_url = urlparse(s3_url)
    if not parsed_s3_url.path:
        raise Exception('invalid S3 URL!')
    if parsed_s3_url.netloc != S3_BUCKET:
        raise Exception('S3 URL {u} must be in pinlogs bucket, but was in {b}'.format(u=s3_url, b=parsed_s3_url.netloc))

    path = parsed_s3_url.path
    if path and path[0] == '/':
        # strip the leading slash, since boto API only returns results if 'prefix' has no leading slash
        path = path[1:]
    return parsed_s3_url.netloc, path


def s3_ls(loc):
    bucket, key = parse_s3_url(loc)
    response = s3_client().list_objects(Bucket=bucket, Prefix=key)
    if 'Contents' not in response:
        return []
    paths = [c['Key'] for c in response['Contents']]
    files = [os.path.join('s3://', S3_BUCKET, p) for p in paths]
    return [f for f in files if '_SUCCESS' not in f]


def s3_read(s3_url, line_mapper=None):
    """Read a file from s3 in streaming fashion.
    The entire file is not downloaded to local disk first (which is particulary nice
    as we will shortly be moving to instances without much local/attached storage.
    """
    fileContents = StringIO()
    bucket, key = parse_s3_url(s3_url)
    s3_client().download_fileobj(Bucket=bucket, Key=key, Fileobj=fileContents)
    # seek back to the start of the in-memory buffer written, before passing the fileobj
    # to be decompressed/read.
    fileContents.seek(0)

    decompressed = fileContents if not s3_url.endswith('.gz') else gzip.GzipFile(fileobj=fileContents, mode='rb')
    lines = decompressed.readlines()
    if line_mapper:
        for line in lines:
            yield line_mapper(line)
    else:
        for line in lines:
            yield line
    fileContents.close()


def s3_get(s3_url, local_path=None):
    """Copy over a file from s3 to the local fs.
    """
    if not local_path:
        local_path = os.path.join(
            FLAGS.local_data_cache, hashlib.md5(s3_url).hexdigest())
    local_file_exists = os.path.isfile(local_path)
    if local_file_exists:
        logger.warning('Deleting {l}'.format(l=local_path),
                       extra={'source': 's3_get'})
        os.remove(local_path)

    bucket, key = parse_s3_url(s3_url)
    with open(local_path, 'wb') as local_fd:
        s3_client().download_fileobj(Bucket=bucket, Key=key, Fileobj=local_fd)

    return local_path


def s3_put(local_path, s3_url):
    """Copy a file from the local fs over to s3.
    """
    log_info('Transfering {l} -> {s}'.format(l=local_path, s=s3_url), 's3_put')
    bucket, key = parse_s3_url(s3_url)
    with open(local_path, 'rb') as local:
        s3_client().upload_fileobj(Fileobj=local, Bucket=bucket, Key=key)


def s3_put_numpy_array(arr, s3_path):
    """Serialize a numpy array to s3 file.
    """
    s3_comps = s3_path.split('/')
    fname_prefix = s3_comps[-1] or s3_comps[-2]
    local_path = os.path.join(FLAGS.local_data_cache, fname_prefix)
    np.save(local_path, arr)
    if local_path.endswith('.npy'):
        full_local_path = local_path
    else:
        full_local_path = local_path + '.npy'
    s3_put(full_local_path, s3_path)
    os.remove(full_local_path)


def s3_get_numpy_array(s3_path):
    """Deserialize a numpy array from a s3 file.
    """
    local_path = os.path.join(FLAGS.local_data_cache, s3_path.split('/')[-1]) + '.npy'
    local_path = s3_get(s3_path, local_path)
    arr = np.load(local_path)
    os.remove(local_path)
    return arr


def s3_put_hive_table(ids, arr, s3_path, num_threads=8, chunk_size=150000, typ=np.float16):
    """Write numpy array in a format easily consumable from hive.
    Each row is of the form:
      id\tbase-64-encoded-numpy-serialized-data
    """
    idx_and_ids = [x for x in enumerate(ids)]
    random.shuffle(idx_and_ids)
    chunks = (idx_and_ids[i:i+chunk_size] for i in xrange(0, len(ids), chunk_size))
    id_and_chunks = [x for x in enumerate(chunks)]
    s3_comps = s3_path.split('/')
    fname_prefix = s3_comps[-1] or s3_comps[-2]

    def process_chunk(id_and_chunk):
        chunk_id, chunk = id_and_chunk
        chunk_fname = str.format('{0:06d}.gz', chunk_id)
        fname = os.path.join(
            FLAGS.local_data_cache, fname_prefix + '_' + chunk_fname)
        with gzip.open(fname, 'w') as f:
            for idx, object_id in chunk:
                row = arr[idx]
                row_str = base64.b64encode(row.astype(typ).tostring())
                line = '{i}\t{row}\n'.format(i=object_id, row=row_str)
                f.write(line)
        s3_put(fname, os.path.join(s3_path, chunk_fname))
        os.remove(fname)

    pool = ThreadPool(num_threads)
    pool.map(process_chunk, id_and_chunks)
    pool.close()
    pool.join()


def s3_get_hive_table(ids, dim, s3_path, num_threads=8, init=None, typ=np.float16):
    """Read an array previously serialized using s3_put_hive_table
    """
    log_info('Initializing array of dimension: {i} x {d}'.format(i=len(ids), d=dim),
             'get_hive_table')
    arr = init or np.zeros([len(ids), dim], dtype=np.float32)
    s3_files = s3_ls(s3_path)
    id2idx = {d: x for x, d in enumerate(ids)}
    log_info('Reading {n} files using {t} threads'.format(n=len(s3_files), t=num_threads),
             'get_hive_table')
    lock = threading.Lock()

    def process_chunk(s3_loc):
        log_info('Reading ' + s3_loc, 'get_hive_table')
        lines = s3_read(s3_loc)
        idx2val = {}
        for line in lines:
            obj_id, obj_val = line.rstrip().split('\t')
            if obj_id in id2idx:
                idx = id2idx[obj_id]
                idx2val[idx] = np.fromstring(base64.b64decode(obj_val), dtype=typ)
        lock.acquire()
        try:
            for idx, val in idx2val.iteritems():
                arr[idx, :] = val
        finally:
            lock.release()
        return len(idx2val)

    pool = ThreadPool(num_threads)
    pool.map(process_chunk, s3_files)
    return arr


def setup_source_queue(loc, num_epochs):
    """Add all the files in a S3 directory to a queue.
    """
    sources = s3_ls(loc)
    q = Queue.Queue()
    for epoch_num in range(num_epochs):
        random.shuffle(sources)
        for source in sources:
            q.put(source)
    return q
