import java.util.Date;
import java.io.Serializable;
import java.text.SimpleDateFormat;

public class Message implements Serializable
{
  private String sender,
                 recipient;
  private String body;
  private Date d;

  //create a new message
  public Message(String sender, String recipient, String body)
  {
    this.sender     = sender;
    this.recipient  = recipient;
    this.body       = body;
    d               = new Date();
  }

  public Message(String recipient)
  {
    this ("", recipient, "");
  }

  public String getSender()
  {
    return this.sender;
  }

  public String getRecipient()
  {
    return this.recipient;
  }

  public String getBody()
  {
    return this.body;
  }

  @Override
  public String toString()
  {
    return 
      "\nFrom: " + sender +
      "\nTo: " + recipient +
      "\nOn: " + new SimpleDateFormat("dd/MM/yyyy HH:mm").format(d) +
      "\nMessage:\n" + body;
  }
}
