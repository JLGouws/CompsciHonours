import java.io.Serializable;

public class Message implements Serializable
{
  private String sender,
                 recipient;
  private String body;

  public Message(String sender, String recipient, String body)
  {
    this.sender     = sender;
    this.recipient  = recipient;
    this.body       = body;
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
      "\nMessage:\n" + body;
  }
}
