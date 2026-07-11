#include "kernels__macros.h"
#include "kernels.h"

// _o = observer
// _i = incident light

// Grey-scale singlle colour channel functons:

// D - Trowbridge-Reitz approximation
float Distribution_of_facetes( float3 omega_h, float alpha, float3 surface_normal ){											// omega_h=    ,   alpha=roughness, n=surface normal
	float alpha_sq		=	pown(alpha, 2);
	float b				=	pown(( pown( dot(surface_normal, omega_h),2) * (alpha_sq -1)  +1), 2);
	return					alpha_sq / ( M_PI_F * b);
}

// Schlick's fn
float G_Schlick( float cos_omega, float k){
	return cos_omega / mad( cos_omega, (1-k), k);
}

// G - Smiths's Schlick-GGX approximation
float Geometric_attenuation( float omega_o, float omega_i, float3 normal, float roughness ){
	float k				=	pown((roughness + 1), 2) / 8.0f;
	return					G_Schlick( dot(normal, cos(omega_o)), k) *  G_Schlick( dot(normal, cos(omega_i)), k);
}

// Fr - Schick's approximation
float Fresnel_reflectance( float cos_theta, float base_reflectance ){
	return 					(	(1.0f-base_reflectance) * pown( (1-cos_theta), 5)   +   base_reflectance	);
}

// Torrance-Sparrow Micro facet Model
//bdrf( rho, omega_o, omega_i ) = ( D(omega_h) * G(omega_o, omega_i) * Fr( omega_o) ) / ( 4cos( theta_o) cos(theta_i) )

float Cook_Torrance( float alpha, float3 surface_normal,		float omega_o, float omega_i, float roughness,		float base_reflectance, float albedo  ){
	float3	omega_half			=	normalize(		omega_o + omega_i);
	float	len_h				=	fast_length(	omega_half);
	float	len_i				=	fast_length(	omega_i);
	float	len_o				=	fast_length(	omega_o);
	float	len_n				=	fast_length(	surface_normal);

	float	cos_theta_o_h		=	dot(	omega_o,  omega_half )		/ ( len_o * len_h);
	float	cos_theta_o_n		=	dot(	omega_o,  surface_normal )	/ ( len_o * len_n);
	float	cos_theta_i_n		=	dot(	omega_i,  surface_normal )	/ ( len_i * len_n);

	float	D					=	Distribution_of_facetes(	omega_half,		alpha,				surface_normal 					);
	float	G					=	Geometric_attenuation(		omega_o,		omega_i,			surface_normal,		roughness	);
	float	F					=	Fresnel_reflectance(		cos_theta_o_h,	base_reflectance									);
	float	K_d					=	1.0f - F;

	float	Lambertian			=	(K_d * albedo/M_1_PI_F );
	float	Torrance_Sparrow	=	(D*G*F)/(4 * cos_theta_o_n * cos_theta_i_n );
	return						(	Lambertian	+	Torrance_Sparrow );
}

//  Light Transport Equation (LTE)


float3 L_o( float3 omega_o, p){		// where p = point implies a set of light sources

	// NB need Fresnel_reflectance(..) for each light to compute Kd_i

	float3	d_omega_i;			// for each light

	float3	irradiance			=	L_i(p,omega_i) * absdot(omega_i, surface_normal); // for each light

	float3	sum_Lambertian		=	SUM( (Kd_i * Lambertian(..) * irradiance ),  d_omega_i );

	float3	sum_Specular		=	SUM( (Torrance_Sparrow(..)	* irradiance ),  d_omega_i );

	return	(sum_Lambertian + sum_Specular);
}



// Spherical harmonic illuminaton, for diffuse reflection. For specular reflection, need to integrate the illumination fn with the speularity fn.... esp roughness.
