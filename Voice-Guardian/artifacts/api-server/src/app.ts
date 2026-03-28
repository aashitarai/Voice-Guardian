import express, { type Express } from "express";
import cors from "cors";
import pinoHttp from "pino-http";
import { createProxyMiddleware } from "http-proxy-middleware";
import router from "./routes";
import { logger } from "./lib/logger";

const app: Express = express();

app.use(
  pinoHttp({
    logger,
    serializers: {
      req(req) {
        return {
          id: req.id,
          method: req.method,
          url: req.url?.split("?")[0],
        };
      },
      res(res) {
        return {
          statusCode: res.statusCode,
        };
      },
    },
  }),
);
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use("/api", router);

app.use(
  "/",
  createProxyMiddleware({
    target: "http://localhost:5000",
    changeOrigin: true,
    ws: true,
    on: {
      error: (err, _req, res) => {
        logger.warn({ err }, "Streamlit proxy error");
        if (res && "writeHead" in res) {
          (res as import("http").ServerResponse).writeHead(502);
          (res as import("http").ServerResponse).end(
            "Digital Guardian is starting up, please wait...",
          );
        }
      },
    },
  }),
);

export default app;
