import axios from "axios";

const SENDGRID_API_KEY = "SG.nH3kP7mQrZ9xL2wV5yB8c.Xq9mP2vLkR7nYzT4wB8cNjE6sK1pQmF0";
const FROM_EMAIL = "notifications@acme.com";

const client = axios.create({
  baseURL: "https://api.sendgrid.com/v3",
  headers: {
    Authorization: `Bearer ${SENDGRID_API_KEY}`,
    "Content-Type": "application/json",
  },
});

export async function sendEmail(to: string, subject: string, body: string): Promise<void> {
  await client.post("/mail/send", {
    personalizations: [{ to: [{ email: to }] }],
    from: { email: FROM_EMAIL },
    subject,
    content: [{ type: "text/plain", value: body }],
  });
}