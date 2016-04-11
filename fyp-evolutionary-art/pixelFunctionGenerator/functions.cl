float noise( float2 x ) {
	return sin(1.5f*x.x)*sin(1.5f*x.y);
}

float2 multiply2x2(float2 p, float2 row0, float2 row1) {
	return (float2)(dot(row0, p), dot(row1, p));
}

float fbm4(float x, float y) {
	float2 p = (float2)(x, y);
	float f = 0.0f;
	f += 0.5000f*noise(p); p = multiply2x2(p, (float2)(2.0f, 1.2f), (float2)(-1.2f, 2.0f));
	f += 0.2500f*noise(p); p = multiply2x2(p, (float2)(2.0f, 1.2f), (float2)(-1.2f, 2.0f));
	f += 0.1250f*noise(p); p = multiply2x2(p, (float2)(2.0f, 1.2f), (float2)(-1.2f, 2.0f));
	f += 0.0625f*noise(p);
	return f/0.9375f;
}

float fbm6(float x, float y) {
	float2 p = (float2)(x, y);
	float f = 0.0f;
	f += 0.500000f*(noise( p )); p = multiply2x2(p, (float2)(2.0f, 1.2f), (float2)(-1.2f, 2.0f));
	f += 0.250000f*(noise( p )); p = multiply2x2(p, (float2)(2.0f, 1.2f), (float2)(-1.2f, 2.0f));
	f += 0.125000f*(noise( p )); p = multiply2x2(p, (float2)(2.0f, 1.2f), (float2)(-1.2f, 2.0f));
	f += 0.062500f*(noise( p )); p = multiply2x2(p, (float2)(2.0f, 1.2f), (float2)(-1.2f, 2.0f));
	f += 0.031250f*(noise( p )); p = multiply2x2(p, (float2)(2.0f, 1.2f), (float2)(-1.2f, 2.0f));
	f += 0.015625f*(noise( p ));
	return f/0.96875f;
}

float hash(float a, float b) {
    int2 c = (int2)(as_int(a), as_int(b));
    int x = 0x3504f333*c.x*c.x + c.y;
    int y = 0xf1bbcdcb*c.y*c.y + c.x;
    
    return float(x*y)*(2.0/8589934592.0)+0.5;
}




