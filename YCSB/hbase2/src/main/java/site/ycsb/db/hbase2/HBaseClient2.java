/**
 * Licensed under the Apache License, Version 2.0 (the "License"); you
 * may not use this file except in compliance with the License. You
 * may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing
 * permissions and limitations under the License. See accompanying
 * LICENSE file.
 */

package site.ycsb.db.hbase2;

import org.apache.hadoop.hbase.CompareOperator;
import org.apache.hadoop.hbase.filter.BinaryComparator;
import org.apache.hadoop.hbase.filter.ByteArrayComparable;
import org.apache.hadoop.hbase.filter.FilterList;
import org.apache.hadoop.hbase.filter.ValueFilter;
import site.ycsb.ByteArrayByteIterator;
import site.ycsb.ByteIterator;
import site.ycsb.DBException;
import site.ycsb.Status;
import site.ycsb.measurements.Measurements;

import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.Cell;
import org.apache.hadoop.hbase.CellUtil;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.TableName;
// import org.apache.hadoop.hbase.client.Admin;
import org.apache.hadoop.hbase.client.BufferedMutator;
import org.apache.hadoop.hbase.client.BufferedMutatorParams;
import org.apache.hadoop.hbase.client.Connection;
import org.apache.hadoop.hbase.client.ConnectionFactory;
import org.apache.hadoop.hbase.client.Delete;
import org.apache.hadoop.hbase.client.Durability;
import org.apache.hadoop.hbase.client.Get;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.client.ResultScanner;
import org.apache.hadoop.hbase.client.Scan;
import org.apache.hadoop.hbase.client.Table;
import org.apache.hadoop.hbase.filter.PageFilter;
import org.apache.hadoop.hbase.util.Bytes;

import java.io.IOException;
import java.util.ConcurrentModificationException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.Vector;
import java.util.concurrent.atomic.AtomicInteger;

/** Added by W.Z. */
import java.util.ArrayList;
import site.ycsb.workloads.TCacheWorkload.TCacheQuery;
/** End of modification by W.Z. */

import static site.ycsb.workloads.CoreWorkload.TABLENAME_PROPERTY;
import static site.ycsb.workloads.CoreWorkload.TABLENAME_PROPERTY_DEFAULT;

/**
 * HBase 2 client for YCSB framework.
 *
 * Intended for use with HBase's shaded client.
 */
public class HBaseClient2 extends site.ycsb.DB {
  private static final AtomicInteger THREAD_COUNT = new AtomicInteger(0);

  private Configuration config = HBaseConfiguration.create();

  private boolean debug = true;

  private String tableName = "";

  /**
   * A Cluster Connection instance that is shared by all running ycsb threads.
   * Needs to be initialized late so we pick up command-line configs if any.
   * To ensure one instance only in a multi-threaded context, guard access
   * with a 'lock' object.
   * @See #CONNECTION_LOCK.
   */
  private static Connection connection = null;

  // Depending on the value of clientSideBuffering, either bufferedMutator
  // (clientSideBuffering) or currentTable (!clientSideBuffering) will be used.
  private Table currentTable = null;
  private BufferedMutator bufferedMutator = null;

  private String columnFamily = "";
  private byte[] columnFamilyBytes;

  /**
   * Durability to use for puts and deletes.
   */
  private Durability durability = Durability.USE_DEFAULT;

  /** Whether or not a page filter should be used to limit scan length. */
  private boolean usePageFilter = true;

  /**
   * If true, buffer mutations on the client. This is the default behavior for
   * HBaseClient. For measuring insert/update/delete latencies, client side
   * buffering should be disabled.
   */
  private boolean clientSideBuffering = false;
  private long writeBufferSize = 1024 * 1024 * 12;

  /**
   * If true, we will configure server-side value filtering during scans.
   */
  private boolean useScanValueFiltering = false;
  private CompareOperator scanFilterOperator;
  private static final String DEFAULT_SCAN_FILTER_OPERATOR = "less_or_equal";
  private ByteArrayComparable scanFilterValue;
  private static final String DEFAULT_SCAN_FILTER_VALUE = // 200 hexadecimal chars translated into 100 bytes
      "7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF" +
      "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF";

  /** Added by W.Z, whether to execute or print queries (transactions). */
  public static final String PRINT_OR_EXE = "DB_PRINT_EXE";
  private String printExe;
  /** End of modification by W.Z. */

  /**
   * Initialize any state for this DB. Called once per DB instance; there is one
   * DB instance per client thread.
   */
  @Override
  public void init() throws DBException {
    if ("true"
        .equals(getProperties().getProperty("clientbuffering", "false"))) {
      this.clientSideBuffering = true;
    }
    if (getProperties().containsKey("writebuffersize")) {
      writeBufferSize =
          Long.parseLong(getProperties().getProperty("writebuffersize"));
    }

    if (getProperties().getProperty("durability") != null) {
      this.durability = Durability.valueOf(getProperties().getProperty("durability"));
    }
    
    /** Added by W.Z to control workload transaction printing. */
    this.printExe = getProperties().getProperty("DB_PRINT_EXE", "print_only");
    /** End of modification by W.Z. */

    if ("kerberos".equalsIgnoreCase(config.get("hbase.security.authentication"))) {
      config.set("hadoop.security.authentication", "Kerberos");
      UserGroupInformation.setConfiguration(config);
    }

    if ((getProperties().getProperty("principal") != null)
        && (getProperties().getProperty("keytab") != null)) {
      try {
        UserGroupInformation.loginUserFromKeytab(getProperties().getProperty("principal"),
              getProperties().getProperty("keytab"));
      } catch (IOException e) {
        System.err.println("Keytab file is not readable or not found");
        throw new DBException(e);
      }
    }

    String table = getProperties().getProperty(TABLENAME_PROPERTY, TABLENAME_PROPERTY_DEFAULT);
    try {
      THREAD_COUNT.getAndIncrement();
      synchronized (THREAD_COUNT) {
        if (connection == null) {
          // Initialize if not set up already.
          // added for debug
          System.out.println("HBase connection config: " + config);
          // End of modification by W.Z
          connection = ConnectionFactory.createConnection(config);
          // added for debug
          System.out.println("HBase connection: " + connection);
          // End of modification by W.Z
          // Terminate right now if table does not exist, since the client
          // will not propagate this error upstream once the workload
          // starts.
          final TableName tName = TableName.valueOf(table);
          // added by W.Z for debug
          System.out.println("getHTable: " + tName);
          // End of modification
          this.currentTable = connection.getTable(tName);
          System.out.println("GET HTable Sccuess.");

          // added by W.Z for get Request Debug
          Result r = null;
          this.columnFamily = "family";
          String key = "xxxx";
          String field = "qual1";
          try {
            System.out
                .println("Doing read from HBase columnfamily " + columnFamily);
            System.out.println("Doing read for key: " + key);
            Get g = new Get(Bytes.toBytes(key));
            g.addColumn(Bytes.toBytes(columnFamily), Bytes.toBytes(field));
            r = currentTable.get(g);
          } catch (IOException e) {
            System.err.println("Error doing get: " + e);
          }
          if (r == null) {
            System.out.println("HBase get result is empty. " + field);
          }
          while (r.advance()) {
            final Cell c = r.current();
            System.out.println(
                "Result for field: " + Bytes.toString(CellUtil.cloneQualifier(c))
                    + " is: " + Bytes.toString(CellUtil.cloneValue(c)));
          }
          // end of modification

          // added by W.Z for put Request debug
          Put p = new Put(Bytes.toBytes(key));
          field = "qual2";
          p.setDurability(durability);
          p.addColumn(Bytes.toBytes(columnFamily), Bytes.toBytes(field), Bytes.toBytes("value2"));
          this.currentTable.put(p);
          try {
            System.out
                .println("Doing read from HBase columnfamily " + columnFamily);
            System.out.println("Doing read for key: " + key);
            Get g = new Get(Bytes.toBytes(key));
            g.addColumn(Bytes.toBytes(columnFamily), Bytes.toBytes(field));
            r = currentTable.get(g);
          } catch (IOException e) {
            System.err.println("Error doing get: " + e);
          }
          if (r == null) {
            System.out.println("HBase get result is empty. " + field);
          }
          while (r.advance()) {
            final Cell c = r.current();
            System.out.println(
                "Result for field: " + Bytes.toString(CellUtil.cloneQualifier(c))
                    + " is: " + Bytes.toString(CellUtil.cloneValue(c)));
          }
          // end of modification

          // try (Admin admin = connection.getAdmin()) {
          //   if (!admin.tableExists(tName)) {
          //     throw new DBException("Table " + tName + " does not exists");
          //   }
          //   // added for debug
          //   System.out.println("HBase admin: " + admin);
          //   // End of modification by W.Z
          // }
        }
      }
    } catch (java.io.IOException e) {
      throw new DBException(e);
    }

    if ((getProperties().getProperty("debug") != null)
        && (getProperties().getProperty("debug").compareTo("true") == 0)) {
      debug = true;
    }

    usePageFilter = isBooleanParamSet("hbase.usepagefilter", usePageFilter);


    if (isBooleanParamSet("hbase.usescanvaluefiltering", false)) {
      useScanValueFiltering=true;
      String operator = getProperties().getProperty("hbase.scanfilteroperator");
      operator = operator == null || operator.trim().isEmpty() ? DEFAULT_SCAN_FILTER_OPERATOR : operator;
      scanFilterOperator = CompareOperator.valueOf(operator.toUpperCase());
      String filterValue = getProperties().getProperty("hbase.scanfiltervalue");
      filterValue = filterValue == null || filterValue.trim().isEmpty() ? DEFAULT_SCAN_FILTER_VALUE : filterValue;
      scanFilterValue = new BinaryComparator(Bytes.fromHex(filterValue));
    }

    columnFamily = getProperties().getProperty("columnfamily");
    if (columnFamily == null) {
      System.err.println("Error, must specify a columnfamily for HBase table");
      throw new DBException("No columnfamily specified");
    }
    columnFamilyBytes = Bytes.toBytes(columnFamily);
  }

  /**
   * Cleanup any state for this DB. Called once per DB instance; there is one DB
   * instance per client thread.
   */
  @Override
  public void cleanup() throws DBException {
    // Get the measurements instance as this is the only client that should
    // count clean up time like an update if client-side buffering is
    // enabled.
    Measurements measurements = Measurements.getMeasurements();
    try {
      long st = System.nanoTime();
      if (bufferedMutator != null) {
        bufferedMutator.close();
      }
      if (currentTable != null) {
        currentTable.close();
      }
      long en = System.nanoTime();
      final String type = clientSideBuffering ? "UPDATE" : "CLEANUP";
      measurements.measure(type, (int) ((en - st) / 1000));
      int threadCount = THREAD_COUNT.decrementAndGet();
      if (threadCount <= 0) {
        // Means we are done so ok to shut down the Connection.
        synchronized (THREAD_COUNT) {
          if (connection != null) {   
            connection.close();   
            connection = null;    
          }   
        }
      }
    } catch (IOException e) {
      throw new DBException(e);
    }
  }

  public void getHTable(String table) throws IOException {
    final TableName tName = TableName.valueOf(table);
    // added by W.Z for debug
    System.out.println("getHTable: " + tName);
    // End of modification
    this.currentTable = connection.getTable(tName);
    // added for debug
    System.out.println("GET HTable Sccuess. clientSideBuffering: " + clientSideBuffering);
    // End of modification by W.Z
    if (clientSideBuffering) {
      final BufferedMutatorParams p = new BufferedMutatorParams(tName);
      p.writeBufferSize(writeBufferSize);
      this.bufferedMutator = connection.getBufferedMutator(p);
    }
  }

  
  /**
   * Added by W.Z. Support read-only, write-only transactions.
   */
  @Override
  public Status readtxn(ArrayList<TCacheQuery> readlist) {
    if (printExe.equals("print_only")) {
      System.out.println("Transaction Start");
    }
    for (TCacheQuery rQuery : readlist) {
      if (printExe.equals("print_only")) {
        String rQueryStr = String.format("GET KEY: %s FIELDS: %s", rQuery.getKeyName(),
            rQuery.getFields().toString());
        System.out.println(rQueryStr);
      } else {
        this.read(rQuery.getTable(), rQuery.getKeyName(), rQuery.getFields(), rQuery.getValues());
      }
    }
    if (printExe.equals("print_only")) {
      System.out.println("Transaction End");
    }
    // End of modification by W.Z
    return Status.OK;
  }

  @Override
  public Status writetxn(ArrayList<TCacheQuery> writelist) {
    if (printExe.equals("print_only")) {
      System.out.println("Transaction Start");
    }
    for (TCacheQuery wQuery : writelist) {
      if (printExe.equals("print_only")) {
        // construct the string for qualifier: value
        String qualValStr = "";
        for (Map.Entry<String, ByteIterator> entry : wQuery.getValues().entrySet()) {
          String tmpQualVal = String.format("%s: %s, ", entry.getKey(), entry.getValue().toString());
          qualValStr += tmpQualVal;
        }
        String wQueryStr = String.format("PUT KEY: %s VALUES: %s", wQuery.getKeyName(), qualValStr);
        System.out.println(wQueryStr);
      } else {
        this.update(wQuery.getTable(), wQuery.getKeyName(), wQuery.getValues());
      }
    }
    if (printExe.equals("print_only")) {
      System.out.println("Transaction End");
    }
    return Status.OK;
  }
  /** End of modification by W.Z. */

  
  /**
   * Read a record from the database. Each field/value pair from the result will
   * be stored in a HashMap.
   *
   * @param table
   *          The name of the table
   * @param key
   *          The record key of the record to read.
   * @param fields
   *          The list of fields to read, or null for all of them
   * @param result
   *          A HashMap of field/value pairs for the result
   * @return Zero on success, a non-zero error code on error
   */
  public Status read(String table, String key, Set<String> fields,
      Map<String, ByteIterator> result) {
    // if (printExe.equals("print_only")) {
    //   System.out.printf("HBase read table: %s, key: %s, fields: %s\n", table, key, fields.toString());
    //   return Status.OK;
    // }
    // if this is a "new" table, init HTable object. Else, use existing one
    if (!tableName.equals(table)) {
      currentTable = null;
      try {
        getHTable(table);
        tableName = table;
      } catch (IOException e) {
        System.err.println("Error accessing HBase table: " + e);
        return Status.ERROR;
      }
    }

    Result r = null;
    try {
      if (debug) {
        System.out
            .println("Doing read from HBase columnfamily " + columnFamily);
        System.out.println("Doing read for key: " + key);
      }
      Get g = new Get(Bytes.toBytes(key));
      if (fields == null) {
        g.addFamily(columnFamilyBytes);
      } else {
        for (String field : fields) {
          g.addColumn(columnFamilyBytes, Bytes.toBytes(field));
        }
      }
      r = currentTable.get(g);
    } catch (IOException e) {
      if (debug) {
        System.err.println("Error doing get: " + e);
      }
      return Status.ERROR;
    } catch (ConcurrentModificationException e) {
      // do nothing for now...need to understand HBase concurrency model better
      return Status.ERROR;
    }

    if (r.isEmpty()) {
      return Status.NOT_FOUND;
    }

    while (r.advance()) {
      final Cell c = r.current();
      result.put(Bytes.toString(CellUtil.cloneQualifier(c)),
          new ByteArrayByteIterator(CellUtil.cloneValue(c)));
      if (debug) {
        System.out.println(
            "Result for field: " + Bytes.toString(CellUtil.cloneQualifier(c))
                + " is: " + Bytes.toString(CellUtil.cloneValue(c)));
      }
    }
    return Status.OK;
  }

  /**
   * Perform a range scan for a set of records in the database. Each field/value
   * pair from the result will be stored in a HashMap.
   *
   * @param table
   *          The name of the table
   * @param startkey
   *          The record key of the first record to read.
   * @param recordcount
   *          The number of records to read
   * @param fields
   *          The list of fields to read, or null for all of them
   * @param result
   *          A Vector of HashMaps, where each HashMap is a set field/value
   *          pairs for one record
   * @return Zero on success, a non-zero error code on error
   */
  @Override
  public Status scan(String table, String startkey, int recordcount,
      Set<String> fields, Vector<HashMap<String, ByteIterator>> result) {
    // if this is a "new" table, init HTable object. Else, use existing one
    // if (printExe.equals("print_only")) {
    //   System.out.printf("HBase scan table: %s, key: %s, fields: %s\n", table, startkey, fields.toString());
    //   return Status.OK;
    // }
    if (!tableName.equals(table)) {
      currentTable = null;
      try {
        getHTable(table);
        tableName = table;
      } catch (IOException e) {
        System.err.println("Error accessing HBase table: " + e);
        return Status.ERROR;
      }
    }

    Scan s = new Scan(Bytes.toBytes(startkey));
    // HBase has no record limit. Here, assume recordcount is small enough to
    // bring back in one call.
    // We get back recordcount records
    FilterList filterList = new FilterList(FilterList.Operator.MUST_PASS_ALL);

    s.setCaching(recordcount);
    if (this.usePageFilter) {
      filterList.addFilter(new PageFilter(recordcount));
    }

    // add specified fields or else all fields
    if (fields == null) {
      s.addFamily(columnFamilyBytes);
    } else {
      for (String field : fields) {
        s.addColumn(columnFamilyBytes, Bytes.toBytes(field));
      }
    }

    // define value filter if needed
    if (useScanValueFiltering){
      filterList.addFilter(new ValueFilter(scanFilterOperator, scanFilterValue));
    }

    s.setFilter(filterList);

    // get results
    ResultScanner scanner = null;
    try {
      scanner = currentTable.getScanner(s);
      int numResults = 0;
      for (Result rr = scanner.next(); rr != null; rr = scanner.next()) {
        // get row key
        String key = Bytes.toString(rr.getRow());

        if (debug) {
          System.out.println("Got scan result for key: " + key);
        }

        HashMap<String, ByteIterator> rowResult =
            new HashMap<String, ByteIterator>();

        while (rr.advance()) {
          final Cell cell = rr.current();
          rowResult.put(Bytes.toString(CellUtil.cloneQualifier(cell)),
              new ByteArrayByteIterator(CellUtil.cloneValue(cell)));
        }

        // add rowResult to result vector
        result.add(rowResult);
        numResults++;

        // PageFilter does not guarantee that the number of results is <=
        // pageSize, so this
        // break is required.
        if (numResults >= recordcount) {// if hit recordcount, bail out
          break;
        }
      } // done with row
    } catch (IOException e) {
      if (debug) {
        System.out.println("Error in getting/parsing scan result: " + e);
      }
      return Status.ERROR;
    } finally {
      if (scanner != null) {
        scanner.close();
      }
    }

    return Status.OK;
  }

  /**
   * Update a record in the database. Any field/value pairs in the specified
   * values HashMap will be written into the record with the specified record
   * key, overwriting any existing values with the same field name.
   *
   * @param table
   *          The name of the table
   * @param key
   *          The record key of the record to write
   * @param values
   *          A HashMap of field/value pairs to update in the record
   * @return Zero on success, a non-zero error code on error
   */
  @Override
  public Status update(String table, String key,
      Map<String, ByteIterator> values) {
    // if (printExe.equals("print_only")) {
    //   System.out.printf("HBase update table: %s, key: %s, values: %s\n", table, key, values.toString());
    //   return Status.OK;
    // }
    // if this is a "new" table, init HTable object. Else, use existing one
    if (!tableName.equals(table)) {
      currentTable = null;
      try {
        getHTable(table);
        tableName = table;
      } catch (IOException e) {
        System.err.println("Error accessing HBase table: " + e);
        return Status.ERROR;
      }
    }

    if (debug) {
      System.out.println("Setting up put for key: " + key);
    }
    Put p = new Put(Bytes.toBytes(key));
    p.setDurability(durability);
    for (Map.Entry<String, ByteIterator> entry : values.entrySet()) {
      byte[] value = entry.getValue().toArray();
      if (debug) {
        System.out.println("Adding field/value " + entry.getKey() + "/"
            + Bytes.toStringBinary(value) + " to put request");
      }
      p.addColumn(columnFamilyBytes, Bytes.toBytes(entry.getKey()), value);
    }

    try {
      if (clientSideBuffering) {
        // removed Preconditions.checkNotNull, which throws NPE, in favor of NPE on next line
        bufferedMutator.mutate(p);
      } else {
        currentTable.put(p);
      }
    } catch (IOException e) {
      if (debug) {
        System.err.println("Error doing put: " + e);
      }
      return Status.ERROR;
    } catch (ConcurrentModificationException e) {
      // do nothing for now...hope this is rare
      return Status.ERROR;
    }
    return Status.OK;
  }

  /**
   * Insert a record in the database. Any field/value pairs in the specified
   * values HashMap will be written into the record with the specified record
   * key.
   *
   * @param table
   *          The name of the table
   * @param key
   *          The record key of the record to insert.
   * @param values
   *          A HashMap of field/value pairs to insert in the record
   * @return Zero on success, a non-zero error code on error
   */
  @Override
  public Status insert(String table, String key,
      Map<String, ByteIterator> values) {
    return update(table, key, values);
  }

  /**
   * Delete a record from the database.
   *
   * @param table
   *          The name of the table
   * @param key
   *          The record key of the record to delete.
   * @return Zero on success, a non-zero error code on error
   */
  @Override
  public Status delete(String table, String key) {
    // if (printExe.equals("print_only")) {
    //   System.out.printf("HBase delete table: %s, key: %s\n", table, key);
    //   return Status.OK;
    // }
    // if this is a "new" table, init HTable object. Else, use existing one
    if (!tableName.equals(table)) {
      currentTable = null;
      try {
        getHTable(table);
        tableName = table;
      } catch (IOException e) {
        System.err.println("Error accessing HBase table: " + e);
        return Status.ERROR;
      }
    }

    if (debug) {
      System.out.println("Doing delete for key: " + key);
    }

    final Delete d = new Delete(Bytes.toBytes(key));
    d.setDurability(durability);
    try {
      if (clientSideBuffering) {
        // removed Preconditions.checkNotNull, which throws NPE, in favor of NPE on next line
        bufferedMutator.mutate(d);
      } else {
        currentTable.delete(d);
      }
    } catch (IOException e) {
      if (debug) {
        System.err.println("Error doing delete: " + e);
      }
      return Status.ERROR;
    }

    return Status.OK;
  }

  // Only non-private for testing.
  void setConfiguration(final Configuration newConfig) {
    this.config = newConfig;
  }

  private boolean isBooleanParamSet(String param, boolean defaultValue){
    return Boolean.parseBoolean(getProperties().getProperty(param, Boolean.toString(defaultValue)));
  }

}

/*
 * For customized vim control set autoindent set si set shiftwidth=4
 */