import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.util.UUID;

public class IdentityClient {

    // Ensure your Python server is running on this port!
    private static final String API_URL = "http://127.0.0.1:8000/extract-id-data/";
    
    // We assume Id.jpg is now in the root folder of your project
    private static final String IMAGE_PATH = "C:\\Users\\ehler\\Downloads\\IdentityGuard-KYC\\Id.jpg";

    public static void main(String[] args) {
        System.out.println("--- Sybrin IdentityGuard Java Client (Java 8 Compatible) ---");
        
        File uploadFile = new File(IMAGE_PATH);
        if (!uploadFile.exists()) {
            System.err.println("ERROR: File not found at " + IMAGE_PATH);
            System.err.println("Please copy Id.jpg into your project folder!");
            return;
        }

        try {
            String boundary = UUID.randomUUID().toString();
            URL url = new URL(API_URL);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            
            // Setup the Request
            conn.setDoOutput(true);
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);

            // Construct the Body
            try (OutputStream out = conn.getOutputStream();
                 PrintWriter writer = new PrintWriter(new OutputStreamWriter(out, "UTF-8"), true)) {
                
                // 1. Send Header
                writer.append("--" + boundary).append("\r\n");
                writer.append("Content-Disposition: form-data; name=\"id_card\"; filename=\"Id.jpg\"").append("\r\n");
                writer.append("Content-Type: image/jpeg").append("\r\n");
                writer.append("\r\n").flush();
                
                // 2. Send File Bytes
                Files.copy(uploadFile.toPath(), out);
                out.flush();
                
                // 3. Send Footer
                writer.append("\r\n").flush();
                writer.append("--" + boundary + "--").append("\r\n").flush();
            }

            // Get Response
            int responseCode = conn.getResponseCode();
            System.out.println("Server Response Code: " + responseCode);

            // Read the JSON response
            InputStream resultStream = (responseCode == 200) ? conn.getInputStream() : conn.getErrorStream();
            BufferedReader reader = new BufferedReader(new InputStreamReader(resultStream));
            String line;
            StringBuilder response = new StringBuilder();
            while ((line = reader.readLine()) != null) {
                response.append(line);
            }
            
            System.out.println("AI Output: " + response.toString());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}