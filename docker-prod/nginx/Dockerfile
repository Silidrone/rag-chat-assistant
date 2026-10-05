# Stage 1: build the Vite app
# Vite 8 needs a newer Node than the 20.x other stacks on this host build with.
FROM node:22.17.0 AS build

WORKDIR /usr/src/app

COPY web/package*.json ./
RUN npm ci

COPY web ./
RUN npm run build

# Stage 2: serve it, and proxy the API paths to the Flask container
FROM nginx:latest

RUN rm /etc/nginx/nginx.conf /etc/nginx/conf.d/default.conf
COPY docker-prod/nginx/config/nginx.conf /etc/nginx/nginx.conf
COPY --from=build /usr/src/app/dist /usr/src/app/public

EXPOSE 80
