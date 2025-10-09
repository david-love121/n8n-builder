FROM node:20-alpine
RUN apk add --no-cache curl \ 
  && npm i -g @upstash/context7-mcp@latest \
  && npm cache clean --force

ENV CONTEXT7_PORT=8080
USER node
EXPOSE 8080
CMD ["sh", "-c", "context7-mcp --transport stdio"]
