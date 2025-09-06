public class BMPCaller {
   public static void main(String[] args) {
      BMP2OneByte by = new BMP2OneByte();
      for (int i = 0; i < 7; i++)
      {
         for (int j = 0; j < 6; j++)
         {
            String s = "testCases_" + i + "_" + j;
            by.main(new String[]{s, });

         }
      }
   }
}
