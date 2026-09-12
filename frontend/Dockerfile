# Local/reproducible-run parity only -- the real deploy target for this app
# is Vercel (see ../README.md's Deployment section), which builds natively
# and doesn't use this file at all.
FROM node:20-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:20-slim AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# Baked in at build time, matching Next.js's own constraint that
# NEXT_PUBLIC_* vars are inlined into the client bundle, not read at
# container start -- override via --build-arg for a non-default backend URL.
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL
RUN npm run build

FROM node:20-slim AS run
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public
EXPOSE 3000
CMD ["node", "server.js"]
