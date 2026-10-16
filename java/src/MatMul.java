import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryUsage;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Locale;

/**
 * Basic dense matrix multiplication (triple loop, i-j-k order) in Java.
 *
 * Storage: double[n][n], i.e. an array of n references to separately allocated
 * row arrays of IEEE 754 binary64 values.
 *
 * Usage:
 *   java MatMul selftest
 *   java MatMul bench --n N --data DIR --warmup W --reps R --min-batch-time T [--dump FILE]
 *
 * Bench output (stdout, CSV, one row per warm-up or measured repetition):
 *   language,n,phase,rep,inner,batch_time_s,time_per_mult_s,checksum
 * followed by one comment line "# java_heap_used_bytes,..." after the last repetition.
 */
public final class MatMul {

    /** Results are written here so the JIT cannot discard batched products. */
    static volatile double sink;

    /** Kernel: the timed region is exactly this method (allocation + triple loop). */
    static double[][] multiply(double[][] A, double[][] B, int n) {
        double[][] C = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++) {
                    s += A[i][k] * B[k][j];
                }
                C[i][j] = s;
            }
        }
        return C;
    }

    /** Row-major sequential sum, used to consume the result. */
    static double checksum(double[][] C, int n) {
        double s = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                s += C[i][j];
            }
        }
        return s;
    }

    static double[][] readMatrix(Path path, int n) throws IOException {
        byte[] bytes = Files.readAllBytes(path);
        if (bytes.length != 8L * n * n) {
            throw new IOException("expected " + (8L * n * n) + " bytes in " + path);
        }
        ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        double[][] M = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                M[i][j] = buf.getDouble();
            }
        }
        return M;
    }

    static void writeMatrix(Path path, double[][] M, int n) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(8 * n * n).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                buf.putDouble(M[i][j]);
            }
        }
        try (FileOutputStream out = new FileOutputStream(path.toFile())) {
            out.write(buf.array());
        }
    }

    // ---------- self-test with exact known answers ----------

    static boolean check(String name, boolean ok) {
        System.out.printf("%-28s %s%n", name, ok ? "PASS" : "FAIL");
        return ok;
    }

    static int selftest() {
        boolean ok = true;

        double[][] A3 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        double[][] B3 = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
        double[][] E3 = {{30, 24, 18}, {84, 69, 54}, {138, 114, 90}};
        ok &= check("known 3x3 product", Arrays.deepEquals(multiply(A3, B3, 3), E3));

        double[][] A4 = new double[4][4];
        double[][] I4 = new double[4][4];
        double[][] Z4 = new double[4][4];
        for (int idx = 0; idx < 16; idx++) {
            A4[idx / 4][idx % 4] = idx * 3 - 7;
            I4[idx / 4][idx % 4] = (idx % 5 == 0) ? 1.0 : 0.0;
        }
        ok &= check("A * I == A", Arrays.deepEquals(multiply(A4, I4, 4), A4));
        ok &= check("I * A == A", Arrays.deepEquals(multiply(I4, A4, 4), A4));
        ok &= check("A * 0 == 0", Arrays.deepEquals(multiply(A4, Z4, 4), Z4));

        double[][] A2 = {{1, 2}, {0, 1}};
        double[][] B2 = {{1, 0}, {3, 1}};
        double[][] E2 = {{7, 2}, {3, 1}};
        ok &= check("non-commuting 2x2", Arrays.deepEquals(multiply(A2, B2, 2), E2));

        ok &= check("1x1", Arrays.deepEquals(multiply(new double[][] {{2.5}},
                new double[][] {{-4.0}}, 1), new double[][] {{-10.0}}));

        System.out.println("selftest " + (ok ? "PASSED" : "FAILED"));
        return ok ? 0 : 1;
    }

    // ---------- benchmark ----------

    static int bench(int n, String dataDir, int warmup, int reps, double minBatchTime,
                     String dumpPath) throws IOException {
        double[][] A = readMatrix(Paths.get(dataDir, "A_" + n + ".bin"), n);
        double[][] B = readMatrix(Paths.get(dataDir, "B_" + n + ".bin"), n);
        PrintStream out = System.out;
        out.println("language,n,phase,rep,inner,batch_time_s,time_per_mult_s,checksum");
        out.flush();

        double lastSingle = 0.0;
        for (int w = 0; w < warmup; w++) {
            long t0 = System.nanoTime();
            double[][] C = multiply(A, B, n);
            long t1 = System.nanoTime();
            lastSingle = (t1 - t0) * 1e-9;
            out.printf(Locale.ROOT, "java,%d,warmup,%d,1,%.9e,%.9e,%s%n",
                    n, w, lastSingle, lastSingle, repr(checksum(C, n)));
            out.flush();
        }

        // Batch very short multiplications so each sample lasts >= minBatchTime.
        int inner = 1;
        if (lastSingle > 0.0 && lastSingle < minBatchTime) {
            inner = (int) Math.ceil(minBatchTime / lastSingle);
        }

        double[][] C = null;
        for (int r = 0; r < reps; r++) {
            long t0 = System.nanoTime();
            for (int b = 0; b < inner; b++) {
                C = multiply(A, B, n);
                sink = C[b % n][b % n]; // every result is consumed
            }
            long t1 = System.nanoTime();
            double batch = (t1 - t0) * 1e-9;
            out.printf(Locale.ROOT, "java,%d,measure,%d,%d,%.9e,%.9e,%s%n",
                    n, r, inner, batch, batch / inner, repr(checksum(C, n)));
            out.flush();
        }

        MemoryUsage heap = ManagementFactory.getMemoryMXBean().getHeapMemoryUsage();
        out.printf(Locale.ROOT, "# java_heap_used_bytes,%d,java_heap_committed_bytes,%d,"
                + "java_heap_max_bytes,%d%n", heap.getUsed(), heap.getCommitted(), heap.getMax());

        if (dumpPath != null && C != null) {
            writeMatrix(Paths.get(dumpPath), C, n);
        }
        return 0;
    }

    /** Shortest decimal string that round-trips to the same double. */
    static String repr(double v) {
        return Double.toString(v);
    }

    public static void main(String[] args) throws IOException {
        if (args.length >= 1 && args[0].equals("selftest")) {
            System.exit(selftest());
        }
        if (args.length >= 1 && args[0].equals("bench")) {
            int n = 0, warmup = 3, reps = 5;
            double minBatchTime = 0.1;
            String dataDir = "data", dump = null;
            for (int i = 1; i + 1 < args.length; i += 2) {
                switch (args[i]) {
                    case "--n" -> n = Integer.parseInt(args[i + 1]);
                    case "--data" -> dataDir = args[i + 1];
                    case "--warmup" -> warmup = Integer.parseInt(args[i + 1]);
                    case "--reps" -> reps = Integer.parseInt(args[i + 1]);
                    case "--min-batch-time" -> minBatchTime = Double.parseDouble(args[i + 1]);
                    case "--dump" -> dump = args[i + 1];
                    default -> { }
                }
            }
            if (n <= 0) {
                System.err.println("--n is required");
                System.exit(1);
            }
            System.exit(bench(n, dataDir, warmup, reps, minBatchTime, dump));
        }
        System.err.println("usage: java MatMul selftest | bench --n N [--data DIR] [--warmup W] "
                + "[--reps R] [--min-batch-time T] [--dump FILE]");
        System.exit(1);
    }
}
