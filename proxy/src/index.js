export default {
  async fetch(request, env, ctx) {
    // 1. Handle CORS Preflight (OPTIONS)
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type, Authorization",
        }
      });
    }

    // 2. Parse the target URL
    const url = new URL(request.url);
    const targetUrl = `https://integrate.api.nvidia.com${url.pathname}${url.search}`;

    // 3. Create proxy request, STRIPPING the Origin header
    // NIM rejects non-whitelisted Origins, so we must remove it.
    const newReq = new Request(targetUrl, request);
    newReq.headers.delete("Origin");
    
    try {
      // 4. Fetch from NVIDIA
      const response = await fetch(newReq);
      
      // 5. Append CORS headers to the response
      const newRes = new Response(response.body, response);
      newRes.headers.set("Access-Control-Allow-Origin", "*");
      return newRes;
    } catch (e) {
      return new Response(JSON.stringify({ error: e.message }), {
        status: 500,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Content-Type": "application/json"
        }
      });
    }
  }
};
