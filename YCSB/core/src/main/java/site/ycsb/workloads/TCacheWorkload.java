/**
 * Creating a new YCSB format workload for TCache
 */

package site.ycsb.workloads;

import site.ycsb.*;
import site.ycsb.generator.*;
import site.ycsb.generator.UniformLongGenerator;
import site.ycsb.measurements.Measurements;

import java.io.IOException;
import java.util.*;

import java.io.BufferedReader;
import java.io.FileReader;
// import java.io.File;
// import java.net.URL;

import org.postgresql.Driver;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
// import java.sql.SQLException;



/**
 * Properties of the workload are controlled by parameters assigned at runtime.
 */

public class TCacheWorkload extends Workload {
  /**
   * The name of the database table to run queries against.
   */
  public static final String TABLENAME_PROPERTY = "table";

  /**
   * The default name of the database table to run queries against.
   */
  public static final String TABLENAME_PROPERTY_DEFAULT = "usertable";

  protected String table;

  /**
   * The name of the property for the number of fields in a record.
   */
  public static final String FIELD_COUNT_PROPERTY = "fieldcount";

  /**
   * Default number of fields in a record.
   */
  public static final String FIELD_COUNT_PROPERTY_DEFAULT = "10";
  
  private List<String> fieldnames;

  /**
   * The name of the property for the field length distribution. Options are "uniform", "zipfian"
   * (favouring short records), "constant", and "histogram".
   * <p>
   * If "uniform", "zipfian" or "constant", the maximum field length will be that specified by the
   * fieldlength property. If "histogram", then the histogram will be read from the filename
   * specified in the "fieldlengthhistogram" property.
   */
  public static final String FIELD_LENGTH_DISTRIBUTION_PROPERTY = "fieldlengthdistribution";

  /**
   * The default field length distribution.
   */
  public static final String FIELD_LENGTH_DISTRIBUTION_PROPERTY_DEFAULT = "constant";

  /**
   * The name of the property for the length of a field in bytes.
   */
  public static final String FIELD_LENGTH_PROPERTY = "fieldlength";

  /**
   * The default maximum length of a field in bytes.
   */
  public static final String FIELD_LENGTH_PROPERTY_DEFAULT = "100";

  /**
   * The name of the property for the minimum length of a field in bytes.
   */
  public static final String MIN_FIELD_LENGTH_PROPERTY = "minfieldlength";

  /**
   * The default minimum length of a field in bytes.
   */
  public static final String MIN_FIELD_LENGTH_PROPERTY_DEFAULT = "1";

  /**
   * The name of a property that specifies the filename containing the field length histogram (only
   * used if fieldlengthdistribution is "histogram").
   */
  public static final String FIELD_LENGTH_HISTOGRAM_FILE_PROPERTY = "fieldlengthhistogram";

  /**
   * The default filename containing a field length histogram.
   */
  public static final String FIELD_LENGTH_HISTOGRAM_FILE_PROPERTY_DEFAULT = "hist.txt";

  /**
   * Generator object that produces field lengths.  The value of this depends on the properties that
   * start with "FIELD_LENGTH_".
   */
  protected NumberGenerator fieldlengthgenerator;

  /**
   * The name of the property for deciding whether to read one field (false) or all fields (true) of
   * a record.
   */
  public static final String READ_ALL_FIELDS_PROPERTY = "readallfields";

  /**
   * The default value for the readallfields property.
   */
  public static final String READ_ALL_FIELDS_PROPERTY_DEFAULT = "false";

  protected boolean readallfields;

  /**
   * The name of the property for determining how to read all the fields when readallfields is true.
   * If set to true, all the field names will be passed into the underlying client. If set to false,
   * null will be passed into the underlying client. When passed a null, some clients may retrieve
   * the entire row with a wildcard, which may be slower than naming all the fields.
   */
  public static final String READ_ALL_FIELDS_BY_NAME_PROPERTY = "readallfieldsbyname";

  /**
   * The default value for the readallfieldsbyname property.
   */
  public static final String READ_ALL_FIELDS_BY_NAME_PROPERTY_DEFAULT = "false";

  protected boolean readallfieldsbyname;

  /**
   * The name of the property for deciding whether to write one field (false) or all fields (true)
   * of a record.
   */
  public static final String WRITE_ALL_FIELDS_PROPERTY = "writeallfields";

  /**
   * The default value for the writeallfields property.
   */
  public static final String WRITE_ALL_FIELDS_PROPERTY_DEFAULT = "false";

  protected boolean writeallfields;

  /**
   * The name of the property for deciding whether to check all returned
   * data against the formation template to ensure data integrity.
   */
  public static final String DATA_INTEGRITY_PROPERTY = "dataintegrity";

  /**
   * The default value for the dataintegrity property.
   */
  public static final String DATA_INTEGRITY_PROPERTY_DEFAULT = "false";

  /**
   * Set to true if want to check correctness of reads. Must also
   * be set to true during loading phase to function.
   */
  private boolean dataintegrity;

  /**
   * The name of the property for the proportion of transactions that are reads.
   */
  public static final String READ_PROPORTION_PROPERTY = "readproportion";

  /**
   * The default proportion of transactions that are reads.
   */
  public static final String READ_PROPORTION_PROPERTY_DEFAULT = "0.95";

  /**
   * The name of the property for the proportion of transactions that are updates.
   */
  public static final String UPDATE_PROPORTION_PROPERTY = "updateproportion";

  /**
   * The default proportion of transactions that are updates.
   */
  public static final String UPDATE_PROPORTION_PROPERTY_DEFAULT = "0.05";

  /**
   * The name of the property for the proportion of transactions that are inserts.
   */
  public static final String INSERT_PROPORTION_PROPERTY = "insertproportion";

  /**
   * The default proportion of transactions that are inserts.
   */
  public static final String INSERT_PROPORTION_PROPERTY_DEFAULT = "0.0";

  /**
   * The name of the property for the proportion of transactions that are scans.
   */
  public static final String SCAN_PROPORTION_PROPERTY = "scanproportion";

  /**
   * The default proportion of transactions that are scans.
   */
  public static final String SCAN_PROPORTION_PROPERTY_DEFAULT = "0.0";

  /**
   * The name of the property for the proportion of transactions that are read-modify-write.
   */
  public static final String READMODIFYWRITE_PROPORTION_PROPERTY = "readmodifywriteproportion";

  /**
   * The default proportion of transactions that are scans.
   */
  public static final String READMODIFYWRITE_PROPORTION_PROPERTY_DEFAULT = "0.0";

  /**
   * The name of the property for the the distribution of requests across the keyspace. Options are
   * "uniform", "zipfian" and "latest"
   */
  public static final String REQUEST_DISTRIBUTION_PROPERTY = "requestdistribution";

  /**
   * The default distribution of requests across the keyspace.
   */
  public static final String REQUEST_DISTRIBUTION_PROPERTY_DEFAULT = "uniform";

  /**
   * The name of the property for adding zero padding to record numbers in order to match
   * string sort order. Controls the number of 0s to left pad with.
   */
  public static final String ZERO_PADDING_PROPERTY = "zeropadding";

  /**
   * The default zero padding value. Matches integer sort order
   */
  public static final String ZERO_PADDING_PROPERTY_DEFAULT = "1";


  /**
   * The name of the property for the min scan length (number of records).
   */
  public static final String MIN_SCAN_LENGTH_PROPERTY = "minscanlength";

  /**
   * The default min scan length.
   */
  public static final String MIN_SCAN_LENGTH_PROPERTY_DEFAULT = "1";

  /**
   * The name of the property for the max scan length (number of records).
   */
  public static final String MAX_SCAN_LENGTH_PROPERTY = "maxscanlength";

  /**
   * The default max scan length.
   */
  public static final String MAX_SCAN_LENGTH_PROPERTY_DEFAULT = "1000";

  /**
   * The name of the property for the scan length distribution. Options are "uniform" and "zipfian"
   * (favoring short scans)
   */
  public static final String SCAN_LENGTH_DISTRIBUTION_PROPERTY = "scanlengthdistribution";

  /**
   * The default max scan length.
   */
  public static final String SCAN_LENGTH_DISTRIBUTION_PROPERTY_DEFAULT = "uniform";

  /**
   * The name of the property for the order to insert records. Options are "ordered" or "hashed"
   */
  public static final String INSERT_ORDER_PROPERTY = "insertorder";

  /**
   * Default insert order.
   */
  public static final String INSERT_ORDER_PROPERTY_DEFAULT = "hashed";

  /**
   * Percentage data items that constitute the hot set.
   */
  public static final String HOTSPOT_DATA_FRACTION = "hotspotdatafraction";

  /**
   * Default value of the size of the hot set.
   */
  public static final String HOTSPOT_DATA_FRACTION_DEFAULT = "0.2";

  /**
   * Percentage operations that access the hot set.
   */
  public static final String HOTSPOT_OPN_FRACTION = "hotspotopnfraction";

  /**
   * Default value of the percentage operations accessing the hot set.
   */
  public static final String HOTSPOT_OPN_FRACTION_DEFAULT = "0.8";

  /**
   * How many times to retry when insertion of a single item to a DB fails.
   */
  public static final String INSERTION_RETRY_LIMIT = "core_workload_insertion_retry_limit";
  public static final String INSERTION_RETRY_LIMIT_DEFAULT = "0";

  /**
   * On average, how long to wait between the retries, in seconds.
   */
  public static final String INSERTION_RETRY_INTERVAL = "core_workload_insertion_retry_interval";
  public static final String INSERTION_RETRY_INTERVAL_DEFAULT = "3";

  /**
   * Field name prefix.
   */
  public static final String FIELD_NAME_PREFIX = "fieldnameprefix";

  /**
   * Default value of the field name prefix.
   */
  public static final String FIELD_NAME_PREFIX_DEFAULT = "field";

  protected NumberGenerator keysequence;
  protected DiscreteGenerator operationchooser;
  protected NumberGenerator keychooser;
  protected NumberGenerator fieldchooser;
  protected AcknowledgedCounterGenerator transactioninsertkeysequence;
  protected NumberGenerator scanlength;
  protected boolean orderedinserts;
  protected long fieldcount;
  protected long recordcount;
  protected int zeropadding;
  protected int insertionRetryLimit;
  protected int insertionRetryInterval;

  private Measurements measurements = Measurements.getMeasurements();

  
  // added by W.Z to enable multi-size READ and UPDATE
  /**
   * Inner class for TCache Read/Write Query.
   */
  public class TCacheQuery {
    private String table; // table name
    private long keynum; // record keynum, identify unique query
    private String keyname; // record keyname, generated using keynum
    private HashSet<String> fields; // fields to access for read query
    private HashMap<String, ByteIterator> values; // values to update for write query

    /**
     * Constructor for TCacheQuery.
     * @param keynum  record keynum.
     * @param wrtflag true if this is a write query (update operation)
     */
    public TCacheQuery(String table, long keynum, boolean wrtflag) {
      this.table = table;
      this.keynum = keynum;
      this.keyname = CoreWorkload.buildKeyName(keynum, zeropadding, orderedinserts);
      int querysize = 1;
      if (loadfromdbprop) {
        querysize = getQuerySizeDB(keynum);
      } else if (wrtflag) {
        querysize = 1;  // set write query size to 1 for debug testing
      } else {
        querysize = sizeTable.get(keynum);
      }
      if (wrtflag) {
        this.values = buildMultiValue(keyname, querysize);
      } else {
        this.fields = new HashSet<String>();
        Iterator<String> fieldnamesIterator = fieldnames.iterator();
        int keyFieldCnt = 0;
        while (fieldnamesIterator.hasNext() && keyFieldCnt < querysize) {
          fields.add(fieldnamesIterator.next());
          keyFieldCnt++;
        }
      }
    }

    public String getTable() {
      return this.table;
    }

    public long getKeyNum() {
      return this.keynum;
    }

    public String getKeyName() {
      return this.keyname;
    }

    public HashSet<String> getFields() {
      return this.fields;
    }

    public HashMap<String, ByteIterator> getValues() {
      return this.values;
    }
  }
  
  /**
   * The name of the property for deciding whether to read one field (false) or all fields (true) of
   * a record.
   */
  public static final String MULTI_SIZE_PROPERTY = "multisizeprop";
  public static final String MULTI_SIZE_PROPERTY_DEFAULT = "true";
  private boolean multisizeprop;

  /**
   * The name of the property for deciding whether to use synthetic datasets.
   */
  public static final String SYNTH_DATASET_PROPERTY = "syntheticprop";
  public static final String SYNTH_DATASET_PROPERTY_DEFAULT = "false";
  private boolean syntheticprop;
  
  /**
   * The name of the property for deciding whether to load query and transaction info from database.
   */
  public static final String LOAD_FROM_DB_PROPERTY = "loadfromdbprop";
  public static final String LOAD_FROM_DB_PROPERTY_DEFAULT = "false";
  private boolean loadfromdbprop;
  private static Connection pgConnection;
  private static Driver pgDriver;

  /**
   * Connection properties for loading query and transaction info from database.
   */
  public static final String DB_CONNECTION_URL = "postgres.url";
  public static final String DB_CONNECTION_URL_DEFAULT = "jdbc:postgresql://localhost:5432/ycsb-param";
  public static final String DB_CONNECTION_USER = "postgres.user";
  public static final String DB_CONNECTION_USER_DEFAULT = "postgres";
  public static final String DB_CONNECTION_PASSWD = "postgres.passwd";
  public static final String DB_CONNECTION_PASSWD_DEFAULT = "postgres";
  public static final String DB_QUERY_TABLE = "postgres.querytable";
  public static final String DB_QUERY_TABLE_DEFAULT = "querytable";
  public static final String DB_TXN_TABLE = "postgres.txntable";
  public static final String DB_TXN_TABLE_DEFAULT = "txntable";
  private String dburlString; // postgres connection url
  private String dbusrString; // postgres username
  private String dbpwdString; // postgres password
  private String dbqryTable; // postgres query info table
  private String dbtxnTable;  // postgres transaction info table
  
  /**
   * The name of the property that specifies the data set directory.
   */
  public static final String DATA_SET_DIR_PROPERTY = "datasetdir";
  public static final String DATA_SET_DIR_PROPERTY_DEFAULT = "datasets/";
  private String datasetdir;
  
  /**
   * The name of the property that specifies the filename containing the query size information.
   * 
   */
  public static final String QUERY_SIZE_FILE_PROPERTY = "querysizefile";
  public static final String QUERY_SIZE_FILE_PROPERTY_DEFAULT = "querysize.txt";
  private String querysizefile;

  /**
   * Query size table, referenced by keynum (starting from 0).
   */
  private HashMap<Long, Integer> sizeTable;

  /**
   * Transaction size and read/write flag specified in external file.
   */
  public static final String TXN_SIZE_FILE_PROPERTY = "txnsizefile";
  public static final String TXN_SIZE_FILE_PROPERTY_DEFAULT = "transactionsize.txt";
  private String txnsizefile;

  /**
   * \theta parameter for Coreworkloads using Zipfian request destribution.
   */
  public static final String ZIPFIAN_CONSTANT_PROPERTY = "zipfianconstant";
  public static final String ZIPFIAN_CONSTANT_PROPERTY_DEFAULT = "0.99";
  private double zipfianconstant;

  /**
   * Print keynum for synthetic TCache testing.
   */
  public static final String PRINT_KEYNUM_PROPERTY = "printkeynum";
  public static final String PRINT_KEYNUM_PROPERTY_DEFAULT = "true";
  private boolean printkeynum;

  /**
   * Load MultiSize data from querysizefile.
   * @param QuerySizefile
   * @throws IOException
   */
  private static HashMap<Long, Integer> multiSize(String querysizefile) throws IOException {
    try (BufferedReader in = new BufferedReader(new FileReader(querysizefile))) {
      String str;
      String[] line;
      HashMap<Long, Integer> multiSizeTable = new HashMap<>();
      str = in.readLine();
      if (str == null) {
        throw new IOException("MultiSize Empty input file!\n");
      }
      do {
        // line[0]: keynum, line[1]: value (attribute number representing query size)
        line = str.split("\t");
        multiSizeTable.put(Long.parseLong(line[0]), Integer.parseInt(line[1]));
        str = in.readLine();
      } while (str != null);
      // print sizeTable.size() for debugging.
      System.out.println("multiSizeTable.size(): " + multiSizeTable.size());
      return multiSizeTable;
    } catch (Exception e) {
      System.out.println("Exception: " + e);
    }
    return null;
  }

  /**
   * Get query size at runtime from database.
   * @param keynum query id.
   */
  private Integer getQuerySizeDB(Long keynum) {
    try {
      String qryStr = "SELECT qsize FROM " + dbqryTable + " WHERE keynum = " + keynum.toString() + ";";
      PreparedStatement getQuerySizeSt = pgConnection.prepareStatement(qryStr);
      ResultSet resultSet = getQuerySizeSt.executeQuery();
      int qsize = Integer.parseInt(resultSet.getString(1));
      return qsize;
    } catch (Exception e) {
      System.out.println("Error when getting query size from database keynum: " + keynum);
      System.out.println("Exception content: " + e);
      return null;
    }
  }

  /**
   * Called once in init() by Main thread.
   * @param p property parameter.
   */
  private void modPropInit(Properties p) {
    // initialize properties based on `workload` file configuration
    zipfianconstant = 
        Double.parseDouble(p.getProperty(ZIPFIAN_CONSTANT_PROPERTY, ZIPFIAN_CONSTANT_PROPERTY_DEFAULT));
    printkeynum = 
        Boolean.parseBoolean(p.getProperty(PRINT_KEYNUM_PROPERTY, PRINT_KEYNUM_PROPERTY_DEFAULT));
    loadfromdbprop = Boolean.parseBoolean(p.getProperty(LOAD_FROM_DB_PROPERTY, LOAD_FROM_DB_PROPERTY_DEFAULT));
    dburlString = p.getProperty(DB_CONNECTION_URL, DB_CONNECTION_URL_DEFAULT);
    dbusrString = p.getProperty(DB_CONNECTION_USER, DB_CONNECTION_USER_DEFAULT);
    dbpwdString = p.getProperty(DB_CONNECTION_PASSWD, DB_CONNECTION_PASSWD_DEFAULT);
    dbqryTable = p.getProperty(DB_QUERY_TABLE, DB_QUERY_TABLE_DEFAULT);
    dbtxnTable = p.getProperty(DB_TXN_TABLE, DB_TXN_TABLE_DEFAULT);
    datasetdir = p.getProperty(DATA_SET_DIR_PROPERTY, DATA_SET_DIR_PROPERTY_DEFAULT);
    querysizefile = p.getProperty(QUERY_SIZE_FILE_PROPERTY, QUERY_SIZE_FILE_PROPERTY_DEFAULT);
    querysizefile = datasetdir + querysizefile;
    multisizeprop = Boolean.parseBoolean(p.getProperty(MULTI_SIZE_PROPERTY, MULTI_SIZE_PROPERTY_DEFAULT));
    txnsizefile = p.getProperty(TXN_SIZE_FILE_PROPERTY, TXN_SIZE_FILE_PROPERTY_DEFAULT);
    txnsizefile = datasetdir + txnsizefile;
    syntheticprop = Boolean.parseBoolean(p.getProperty(SYNTH_DATASET_PROPERTY, SYNTH_DATASET_PROPERTY_DEFAULT));
    System.out.println("loadfromdbprop: " + loadfromdbprop);
    System.out.println("multisizeprop: " + multisizeprop);
    System.out.println("syntheticprop: " + syntheticprop);
  }
  
  // modification ended

  public static String buildKeyName(long keynum, int zeropadding, boolean orderedinserts) {
    if (!orderedinserts) {
      keynum = Utils.hash(keynum);
    }
    String value = Long.toString(keynum);
    int fill = zeropadding - value.length();
    String prekey = "user";
    for (int i = 0; i < fill; i++) {
      prekey += '0';
    }
    return prekey + value;
  }

  protected static NumberGenerator getFieldLengthGenerator(Properties p) throws WorkloadException {
    NumberGenerator fieldlengthgenerator;
    String fieldlengthdistribution = p.getProperty(
        FIELD_LENGTH_DISTRIBUTION_PROPERTY, FIELD_LENGTH_DISTRIBUTION_PROPERTY_DEFAULT);
    int fieldlength =
        Integer.parseInt(p.getProperty(FIELD_LENGTH_PROPERTY, FIELD_LENGTH_PROPERTY_DEFAULT));
    int minfieldlength =
        Integer.parseInt(p.getProperty(MIN_FIELD_LENGTH_PROPERTY, MIN_FIELD_LENGTH_PROPERTY_DEFAULT));
    String fieldlengthhistogram = p.getProperty(
        FIELD_LENGTH_HISTOGRAM_FILE_PROPERTY, FIELD_LENGTH_HISTOGRAM_FILE_PROPERTY_DEFAULT);
    if (fieldlengthdistribution.compareTo("constant") == 0) {
      fieldlengthgenerator = new ConstantIntegerGenerator(fieldlength);
    } else if (fieldlengthdistribution.compareTo("uniform") == 0) {
      fieldlengthgenerator = new UniformLongGenerator(minfieldlength, fieldlength);
    } else if (fieldlengthdistribution.compareTo("zipfian") == 0) {
      fieldlengthgenerator = new ZipfianGenerator(minfieldlength, fieldlength);
    } else if (fieldlengthdistribution.compareTo("histogram") == 0) {
      try {
        fieldlengthgenerator = new HistogramGenerator(fieldlengthhistogram);
      } catch (IOException e) {
        throw new WorkloadException(
            "Couldn't read field length histogram file: " + fieldlengthhistogram, e);
      }
    } else {
      throw new WorkloadException(
          "Unknown field length distribution \"" + fieldlengthdistribution + "\"");
    }
    return fieldlengthgenerator;
  }

  
  /**
   * Cleanup the scenario. Called once in the main client thread. Disconnect from PostgreSQL parameter database.
   */
  @Override
  public void cleanup() throws WorkloadException {
    try {
      if (loadfromdbprop) {
        pgConnection.close();
      }
    } catch (Exception e) {
      System.out.println("Error when disconnecting from PostgreSQL workload parameter database: " + e);
    }
  }
  
  
  /**
   * Initialize the scenario.
   * Called once, in the main client thread, before any operations are started.
   */
  @Override
  public void init(Properties p) throws WorkloadException {
    table = p.getProperty(TABLENAME_PROPERTY, TABLENAME_PROPERTY_DEFAULT);
    fieldcount =
        Long.parseLong(p.getProperty(FIELD_COUNT_PROPERTY, FIELD_COUNT_PROPERTY_DEFAULT));
    final String fieldnameprefix = p.getProperty(FIELD_NAME_PREFIX, FIELD_NAME_PREFIX_DEFAULT);
    fieldnames = new ArrayList<>();
    for (int i = 0; i < fieldcount; i++) {
      fieldnames.add(fieldnameprefix + i);
    }
    fieldlengthgenerator = CoreWorkload.getFieldLengthGenerator(p);
    // added by W.Z
    modPropInit(p);   // initialize additional property variables for TCache workload
    if (loadfromdbprop) {   // If load parameter from database, build connection here
      try {
        pgDriver = new Driver();
        Properties connProps = new Properties();
        connProps.setProperty("user", dbusrString);
        connProps.setProperty("password", dbpwdString);
        pgConnection = pgDriver.connect(dburlString, connProps);
        pgConnection.setAutoCommit(true); // each query should be executed at once
      } catch (Exception e) {
        System.out.println("Error while connecting to workload parameter DB: " + e);
      }
    } else {  // Otherwise, load query size from file if using multi-size queries 
      if (multisizeprop) {
        try {
          sizeTable = multiSize(querysizefile);
        } catch (Exception e) {
          System.out.println("multiSize Query File IOException.");
        }
      }
    }
    // end of modification
    recordcount =
        Long.parseLong(p.getProperty(Client.RECORD_COUNT_PROPERTY, Client.DEFAULT_RECORD_COUNT));
    if (recordcount == 0) {
      recordcount = Integer.MAX_VALUE;
    }
    String requestdistrib =
        p.getProperty(REQUEST_DISTRIBUTION_PROPERTY, REQUEST_DISTRIBUTION_PROPERTY_DEFAULT);
    int minscanlength =
        Integer.parseInt(p.getProperty(MIN_SCAN_LENGTH_PROPERTY, MIN_SCAN_LENGTH_PROPERTY_DEFAULT));
    int maxscanlength =
        Integer.parseInt(p.getProperty(MAX_SCAN_LENGTH_PROPERTY, MAX_SCAN_LENGTH_PROPERTY_DEFAULT));
    String scanlengthdistrib =
        p.getProperty(SCAN_LENGTH_DISTRIBUTION_PROPERTY, SCAN_LENGTH_DISTRIBUTION_PROPERTY_DEFAULT);

    long insertstart =
        Long.parseLong(p.getProperty(INSERT_START_PROPERTY, INSERT_START_PROPERTY_DEFAULT));
    long insertcount=
        Integer.parseInt(p.getProperty(INSERT_COUNT_PROPERTY, String.valueOf(recordcount - insertstart)));
    // Confirm valid values for insertstart and insertcount in relation to recordcount
    if (recordcount < (insertstart + insertcount)) {
      System.err.println("Invalid combination of insertstart, insertcount and recordcount.");
      System.err.println("recordcount must be bigger than insertstart + insertcount.");
      System.exit(-1);
    }
    zeropadding =
        Integer.parseInt(p.getProperty(ZERO_PADDING_PROPERTY, ZERO_PADDING_PROPERTY_DEFAULT));

    readallfields = Boolean.parseBoolean(
        p.getProperty(READ_ALL_FIELDS_PROPERTY, READ_ALL_FIELDS_PROPERTY_DEFAULT));
    readallfieldsbyname = Boolean.parseBoolean(
        p.getProperty(READ_ALL_FIELDS_BY_NAME_PROPERTY, READ_ALL_FIELDS_BY_NAME_PROPERTY_DEFAULT));
    writeallfields = Boolean.parseBoolean(
        p.getProperty(WRITE_ALL_FIELDS_PROPERTY, WRITE_ALL_FIELDS_PROPERTY_DEFAULT));

    dataintegrity = Boolean.parseBoolean(
        p.getProperty(DATA_INTEGRITY_PROPERTY, DATA_INTEGRITY_PROPERTY_DEFAULT));
    // Confirm that fieldlengthgenerator returns a constant if data
    // integrity check requested.
    if (dataintegrity && !(p.getProperty(
        FIELD_LENGTH_DISTRIBUTION_PROPERTY,
        FIELD_LENGTH_DISTRIBUTION_PROPERTY_DEFAULT)).equals("constant")) {
      System.err.println("Must have constant field size to check data integrity.");
      System.exit(-1);
    }
    if (dataintegrity) {
      System.out.println("Data integrity is enabled.");
    }

    if (p.getProperty(INSERT_ORDER_PROPERTY, INSERT_ORDER_PROPERTY_DEFAULT).compareTo("hashed") == 0) {
      orderedinserts = false;
    } else {
      orderedinserts = true;
    }
    keysequence = new CounterGenerator(insertstart);
    operationchooser = createOperationGenerator(p);
    transactioninsertkeysequence = new AcknowledgedCounterGenerator(recordcount);
    if (requestdistrib.compareTo("uniform") == 0) {
      keychooser = new UniformLongGenerator(insertstart, insertstart + insertcount - 1);
    } else if (requestdistrib.compareTo("exponential") == 0) {
      double percentile = Double.parseDouble(p.getProperty(
          ExponentialGenerator.EXPONENTIAL_PERCENTILE_PROPERTY,
          ExponentialGenerator.EXPONENTIAL_PERCENTILE_DEFAULT));
      double frac = Double.parseDouble(p.getProperty(
          ExponentialGenerator.EXPONENTIAL_FRAC_PROPERTY,
          ExponentialGenerator.EXPONENTIAL_FRAC_DEFAULT));
      keychooser = new ExponentialGenerator(percentile, recordcount * frac);
    } else if (requestdistrib.compareTo("sequential") == 0) {
      keychooser = new SequentialGenerator(insertstart, insertstart + insertcount - 1);
    } else if (requestdistrib.compareTo("zipfian") == 0) {
      // it does this by generating a random "next key" in part by taking the modulus over the
      // number of keys.
      // If the number of keys changes, this would shift the modulus, and we don't want that to
      // change which keys are popular so we'll actually construct the scrambled zipfian generator
      // with a keyspace that is larger than exists at the beginning of the test. that is, we'll predict
      // the number of inserts, and tell the scrambled zipfian generator the number of existing keys
      // plus the number of predicted keys as the total keyspace. then, if the generator picks a key
      // that hasn't been inserted yet, will just ignore it and pick another key. this way, the size of
      // the keyspace doesn't change from the perspective of the scrambled zipfian generator
      final double insertproportion = Double.parseDouble(
          p.getProperty(INSERT_PROPORTION_PROPERTY, INSERT_PROPORTION_PROPERTY_DEFAULT));
      int opcount = Integer.parseInt(p.getProperty(Client.OPERATION_COUNT_PROPERTY));
      int expectednewkeys = (int) ((opcount) * insertproportion * 2.0); // 2 is fudge factor
      // keychooser = new ScrambledZipfianGenerator(insertstart, insertstart + insertcount + expectednewkeys);
      // modified by W.Z
      keychooser = new ScrambledZipfianGenerator(insertstart, insertstart + insertcount + expectednewkeys,
          zipfianconstant);
      // end of modification
    } else if (requestdistrib.compareTo("latest") == 0) {
      keychooser = new SkewedLatestGenerator(transactioninsertkeysequence);
    } else if (requestdistrib.equals("hotspot")) {
      double hotsetfraction =
          Double.parseDouble(p.getProperty(HOTSPOT_DATA_FRACTION, HOTSPOT_DATA_FRACTION_DEFAULT));
      double hotopnfraction =
          Double.parseDouble(p.getProperty(HOTSPOT_OPN_FRACTION, HOTSPOT_OPN_FRACTION_DEFAULT));
      keychooser = new HotspotIntegerGenerator(insertstart, insertstart + insertcount - 1,
          hotsetfraction, hotopnfraction);
    } else {
      throw new WorkloadException("Unknown request distribution \"" + requestdistrib + "\"");
    }
    fieldchooser = new UniformLongGenerator(0, fieldcount - 1);
    if (scanlengthdistrib.compareTo("uniform") == 0) {
      scanlength = new UniformLongGenerator(minscanlength, maxscanlength);
    } else if (scanlengthdistrib.compareTo("zipfian") == 0) {
      scanlength = new ZipfianGenerator(minscanlength, maxscanlength);
    } else {
      throw new WorkloadException(
          "Distribution \"" + scanlengthdistrib + "\" not allowed for scan length");
    }
    insertionRetryLimit = Integer.parseInt(p.getProperty(
        INSERTION_RETRY_LIMIT, INSERTION_RETRY_LIMIT_DEFAULT));
    insertionRetryInterval = Integer.parseInt(p.getProperty(
        INSERTION_RETRY_INTERVAL, INSERTION_RETRY_INTERVAL_DEFAULT));
  }

  /**
   * Builds a value for a randomly chosen field.
   */
  private HashMap<String, ByteIterator> buildSingleValue(String key) {
    HashMap<String, ByteIterator> value = new HashMap<>();

    String fieldkey = fieldnames.get(fieldchooser.nextValue().intValue());
    ByteIterator data;
    if (dataintegrity) {
      data = new StringByteIterator(buildDeterministicValue(key, fieldkey));
    } else {
      // fill with random data
      data = new RandomByteIterator(fieldlengthgenerator.nextValue().longValue());
    }
    value.put(fieldkey, data);

    return value;
  }

  /**
   * Builds multiple values for selected fields.
   * @param key
   * @return
   */
  private HashMap<String, ByteIterator> buildMultiValue(String key, int size) {
    HashMap<String, ByteIterator> values = new HashMap<>();
    int count = 0;

    for (String fieldkey : fieldnames) {
      if (count >= size) {
        break;
      }
      ByteIterator data;
      if (dataintegrity) {
        data = new StringByteIterator(buildDeterministicValue(key, fieldkey));
      } else {
        // fill with random data
        data = new RandomByteIterator(fieldlengthgenerator.nextValue().longValue());
      }
      values.put(fieldkey, data);
      count++;
    }
    return values;
  }


  /**
   * Builds values for all fields.
   */
  private HashMap<String, ByteIterator> buildValues(String key) {
    HashMap<String, ByteIterator> values = new HashMap<>();

    for (String fieldkey : fieldnames) {
      ByteIterator data;
      if (dataintegrity) {
        data = new StringByteIterator(buildDeterministicValue(key, fieldkey));
      } else {
        // fill with random data
        data = new RandomByteIterator(fieldlengthgenerator.nextValue().longValue());
      }
      values.put(fieldkey, data);
    }
    return values;
  }

  /**
   * Build a deterministic value given the key information.
   */
  private String buildDeterministicValue(String key, String fieldkey) {
    int size = fieldlengthgenerator.nextValue().intValue();
    StringBuilder sb = new StringBuilder(size);
    sb.append(key);
    sb.append(':');
    sb.append(fieldkey);
    while (sb.length() < size) {
      sb.append(':');
      sb.append(sb.toString().hashCode());
    }
    sb.setLength(size);

    return sb.toString();
  }

  /**
   * Do one insert operation. Because it will be called concurrently from multiple client threads,
   * this function must be thread safe. However, avoid synchronized, or the threads will block waiting
   * for each other, and it will be difficult to reach the target throughput. Ideally, this function would
   * have no side effects other than DB operations.
   */
  @Override
  public boolean doInsert(DB db, Object threadstate) {
    int keynum = keysequence.nextValue().intValue();
    String dbkey = CoreWorkload.buildKeyName(keynum, zeropadding, orderedinserts);
    HashMap<String, ByteIterator> values = buildValues(dbkey);

    // added by W.Z for debugging
    // System.out.println("doInsert keynum: " + keynum);
    // modification ended
    
    Status status;
    int numOfRetries = 0;
    do {
      status = db.insert(table, dbkey, values);
      if (null != status && status.isOk()) {
        break;
      }
      // Retry if configured. Without retrying, the load process will fail
      // even if one single insertion fails. User can optionally configure
      // an insertion retry limit (default is 0) to enable retry.
      if (++numOfRetries <= insertionRetryLimit) {
        System.err.println("Retrying insertion, retry count: " + numOfRetries);
        try {
          // Sleep for a random number between [0.8, 1.2)*insertionRetryInterval.
          int sleepTime = (int) (1000 * insertionRetryInterval * (0.8 + 0.4 * Math.random()));
          Thread.sleep(sleepTime);
        } catch (InterruptedException e) {
          break;
        }

      } else {
        System.err.println("Error inserting, not retrying any more. number of attempts: " + numOfRetries +
            "Insertion Retry Limit: " + insertionRetryLimit);
        break;

      }
    } while (true);

    return null != status && status.isOk();
  }

  
  /**
   * Carry out read-only and write-only transactions.
   * @param db Database layer used for testing. Should @Override readtxn and writetxn method.
   * @param threadstate
   * @return false if the workload has nothing more to do. true otherwise.
   */
  // @TODO: make this method Thread safe but avoid synchronized.
  @Override
  public int doOnlyTransactions(DB db, Object threadstate) {
    int qrycount = 0;
    if (syntheticprop) {
      try (BufferedReader in = new BufferedReader(new FileReader(txnsizefile))) {
        String str;
        String[] line;
        str = in.readLine();
        if (str == null) {
          System.out.println("Unexpected txnsizefile exception!");
          throw new IOException("txnsizefile is empty!\n");
        }
        do {
          line = str.split("\t");
          int txnFlag = Integer.parseInt(line[0]);
          String[] tmpTxnCont = line[1].split(",");
          HashSet<Long> tmpkeynumset = new HashSet<Long>();
          for (String keynumstr : tmpTxnCont) {
            tmpkeynumset.add(Long.parseLong(keynumstr));
          }
          ArrayList<TCacheQuery> txnArrayList = new ArrayList<TCacheQuery>();
          // printkeynum for TCache synthetic testing
          if (printkeynum) {
            System.out.println("Keynum Set: " + tmpkeynumset);
          }
          for (long keynum : tmpkeynumset) {
            TCacheQuery tmpquery = new TCacheQuery(table, keynum, txnFlag == 1);
            txnArrayList.add(tmpquery);
          }
          if (txnFlag == 1) {
            db.writetxn(txnArrayList);
          } else { // read-only transaction
            db.readtxn(txnArrayList);
          }
          qrycount += tmpkeynumset.size();
          str = in.readLine();
        } while (str != null);
      } catch (Exception e) {
        System.out.println("Exception: " + e);
      }
    } else {
      try (BufferedReader in = new BufferedReader(new FileReader(txnsizefile))) {
        String str;
        String[] line;
        str = in.readLine();
        if (str == null) {
          System.out.println("Unexpected txnsizefile exception!");
          throw new IOException("txnsizefile is empty!\n");
        }
        do {
          line = str.split("\t");
          int txnFlag = Integer.parseInt(line[0]);
          int txnSize = Integer.parseInt(line[1]);
          HashSet<Long> tmpkeynumset = new HashSet<>();
          while (tmpkeynumset.size() < txnSize) {
            long keynum = nextKeynum();
            tmpkeynumset.add(keynum);
          }
          ArrayList<TCacheQuery> txnArrayList = new ArrayList<TCacheQuery>();
          // printkeynum for TCache synthetic testing
          if (printkeynum) {
            System.out.println("Keynum Set: " + tmpkeynumset);
          }
          for (long keynum : tmpkeynumset) {
            TCacheQuery tmpquery = new TCacheQuery(table, keynum, txnFlag == 1);
            txnArrayList.add(tmpquery);
          }
          if (txnFlag == 1) {
            db.writetxn(txnArrayList);
          } else { // read-only transaction
            db.readtxn(txnArrayList);
          }
          qrycount += tmpkeynumset.size();
          str = in.readLine();
        } while (str != null);
      } catch (Exception e) {
        System.out.println("Exception: " + e);
      }
    }
    return qrycount;
  }
  
  /**
   * Do one transaction operation. Because it will be called concurrently from multiple client
   * threads, this function must be thread safe. However, avoid synchronized, or the threads will block waiting
   * for each other, and it will be difficult to reach the target throughput. Ideally, this function would
   * have no side effects other than DB operations.
   */
  @Override
  public boolean doTransaction(DB db, Object threadstate) {
    String operation = operationchooser.nextString();
    if(operation == null) {
      return false;
    }

    switch (operation) {
    case "READ":
      doTransactionRead(db);
      break;
    case "UPDATE":
      doTransactionUpdate(db);
      break;
    case "INSERT":
      doTransactionInsert(db);
      break;
    case "SCAN":
      doTransactionScan(db);
      break;
    default:
      doTransactionReadModifyWrite(db);
    }

    return true;
  }

  /**
   * Results are reported in the first three buckets of the histogram under
   * the label "VERIFY".
   * Bucket 0 means the expected data was returned.
   * Bucket 1 means incorrect data was returned.
   * Bucket 2 means null data was returned when some data was expected.
   */
  protected void verifyRow(String key, HashMap<String, ByteIterator> cells) {
    Status verifyStatus = Status.OK;
    long startTime = System.nanoTime();
    if (!cells.isEmpty()) {
      for (Map.Entry<String, ByteIterator> entry : cells.entrySet()) {
        if (!entry.getValue().toString().equals(buildDeterministicValue(key, entry.getKey()))) {
          verifyStatus = Status.UNEXPECTED_STATE;
          break;
        }
      }
    } else {
      // This assumes that null data is never valid
      verifyStatus = Status.ERROR;
    }
    long endTime = System.nanoTime();
    measurements.measure("VERIFY", (int) (endTime - startTime) / 1000);
    measurements.reportStatus("VERIFY", verifyStatus);
  }

  long nextKeynum() {
    long keynum;
    if (keychooser instanceof ExponentialGenerator) {
      do {
        keynum = transactioninsertkeysequence.lastValue() - keychooser.nextValue().intValue();
      } while (keynum < 0);
    } else {
      do {
        keynum = keychooser.nextValue().intValue();
      } while (keynum > transactioninsertkeysequence.lastValue());
    }
    return keynum;
  }

  // added by W.Z
  // multisizeprop is true, choose filenames based on determined size
  public void doTransactionRead(DB db) {
    // choose a random key
    long keynum = nextKeynum();

    String keyname = CoreWorkload.buildKeyName(keynum, zeropadding, orderedinserts);

    HashSet<String> fields = null;

    
    if (multisizeprop) {
      // generating multi-size query
      fields = new HashSet<String>();
      Iterator<String> fieldnamesIterator = fieldnames.iterator();
      int keyFieldCnt = 0;
      int keySizeLimit = 1;
      if (loadfromdbprop) {
        keySizeLimit = getQuerySizeDB(keynum);
      } else {
        keySizeLimit = sizeTable.get(keynum);
      }
      while (fieldnamesIterator.hasNext() && keyFieldCnt < keySizeLimit) {
        fields.add(fieldnamesIterator.next());
        keyFieldCnt++;
      } // modification ended
    } else if (!readallfields) {
      // read a random field
      String fieldname = fieldnames.get(fieldchooser.nextValue().intValue());

      fields = new HashSet<String>();
      fields.add(fieldname);
    } else if (dataintegrity || readallfieldsbyname) {
      // pass the full field list if dataintegrity is on for verification
      fields = new HashSet<String>(fieldnames);
    }

    HashMap<String, ByteIterator> cells = new HashMap<String, ByteIterator>();
    db.read(table, keyname, fields, cells);

    if (dataintegrity) {
      verifyRow(keyname, cells);
    }
  }

  public void doTransactionReadModifyWrite(DB db) {
    // choose a random key
    long keynum = nextKeynum();

    String keyname = CoreWorkload.buildKeyName(keynum, zeropadding, orderedinserts);

    HashSet<String> fields = null;

    if (!readallfields) {
      // read a random field
      String fieldname = fieldnames.get(fieldchooser.nextValue().intValue());

      fields = new HashSet<String>();
      fields.add(fieldname);
    }

    HashMap<String, ByteIterator> values;

    if (writeallfields) {
      // new data for all the fields
      values = buildValues(keyname);
    } else {
      // update a random field
      values = buildSingleValue(keyname);
    }

    // do the transaction

    HashMap<String, ByteIterator> cells = new HashMap<String, ByteIterator>();


    long ist = measurements.getIntendedStartTimeNs();
    long st = System.nanoTime();
    db.read(table, keyname, fields, cells);

    db.update(table, keyname, values);

    long en = System.nanoTime();

    if (dataintegrity) {
      verifyRow(keyname, cells);
    }

    measurements.measure("READ-MODIFY-WRITE", (int) ((en - st) / 1000));
    measurements.measureIntended("READ-MODIFY-WRITE", (int) ((en - ist) / 1000));
  }

  public void doTransactionScan(DB db) {
    // choose a random key
    long keynum = nextKeynum();

    String startkeyname = CoreWorkload.buildKeyName(keynum, zeropadding, orderedinserts);

    // choose a random scan length
    int len = scanlength.nextValue().intValue();

    HashSet<String> fields = null;

    if (!readallfields) {
      // read a random field
      String fieldname = fieldnames.get(fieldchooser.nextValue().intValue());

      fields = new HashSet<String>();
      fields.add(fieldname);
    }

    db.scan(table, startkeyname, len, fields, new Vector<HashMap<String, ByteIterator>>());
  }

  // added by W.Z
  // multisizeprop is true, choose filenames based on determined size
  public void doTransactionUpdate(DB db) {
    // choose a random key
    long keynum = nextKeynum();

    String keyname = CoreWorkload.buildKeyName(keynum, zeropadding, orderedinserts);

    HashMap<String, ByteIterator> values;

    
    if (multisizeprop) {
      // generating multi-size query
      int size = 1;
      if (loadfromdbprop) {
        size = getQuerySizeDB(keynum);
      } else {
        size = sizeTable.get(keynum);
      }
      values = buildMultiValue(keyname, size);
      // modification ended
    } else if (writeallfields) {
      // new data for all the fields
      values = buildValues(keyname);
    } else {
      // update a random field
      values = buildSingleValue(keyname);
    }

    db.update(table, keyname, values);
  }

  public void doTransactionInsert(DB db) {
    // choose the next key
    long keynum = transactioninsertkeysequence.nextValue();

    try {
      String dbkey = CoreWorkload.buildKeyName(keynum, zeropadding, orderedinserts);

      HashMap<String, ByteIterator> values = buildValues(dbkey);
      db.insert(table, dbkey, values);
    } finally {
      transactioninsertkeysequence.acknowledge(keynum);
    }
  }

  /**
   * Creates a weighted discrete values with database operations for a workload to perform.
   * Weights/proportions are read from the properties list and defaults are used
   * when values are not configured.
   * Current operations are "READ", "UPDATE", "INSERT", "SCAN" and "READMODIFYWRITE".
   *
   * @param p The properties list to pull weights from.
   * @return A generator that can be used to determine the next operation to perform.
   * @throws IllegalArgumentException if the properties object was null.
   */
  protected static DiscreteGenerator createOperationGenerator(final Properties p) {
    if (p == null) {
      throw new IllegalArgumentException("Properties object cannot be null");
    }
    final double readproportion = Double.parseDouble(
        p.getProperty(READ_PROPORTION_PROPERTY, READ_PROPORTION_PROPERTY_DEFAULT));
    final double updateproportion = Double.parseDouble(
        p.getProperty(UPDATE_PROPORTION_PROPERTY, UPDATE_PROPORTION_PROPERTY_DEFAULT));
    final double insertproportion = Double.parseDouble(
        p.getProperty(INSERT_PROPORTION_PROPERTY, INSERT_PROPORTION_PROPERTY_DEFAULT));
    final double scanproportion = Double.parseDouble(
        p.getProperty(SCAN_PROPORTION_PROPERTY, SCAN_PROPORTION_PROPERTY_DEFAULT));
    final double readmodifywriteproportion = Double.parseDouble(p.getProperty(
        READMODIFYWRITE_PROPORTION_PROPERTY, READMODIFYWRITE_PROPORTION_PROPERTY_DEFAULT));

    final DiscreteGenerator operationchooser = new DiscreteGenerator();
    if (readproportion > 0) {
      operationchooser.addValue(readproportion, "READ");
    }

    if (updateproportion > 0) {
      operationchooser.addValue(updateproportion, "UPDATE");
    }

    if (insertproportion > 0) {
      operationchooser.addValue(insertproportion, "INSERT");
    }

    if (scanproportion > 0) {
      operationchooser.addValue(scanproportion, "SCAN");
    }

    if (readmodifywriteproportion > 0) {
      operationchooser.addValue(readmodifywriteproportion, "READMODIFYWRITE");
    }
    return operationchooser;
  }
}
