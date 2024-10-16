//! Derive proc-macros for `wgcore`.

extern crate proc_macro;

use darling::util::PathList;
use darling::{FromDeriveInput, FromField};
use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{Data, DataStruct, Path};

#[derive(FromDeriveInput, Clone)]
#[darling(attributes(shader))]
struct DeriveShadersArgs {
    #[darling(default)]
    pub derive: PathList,
    #[darling(default)]
    pub composable: Option<bool>,
    pub src: String,
    #[darling(default)]
    pub src_fn: Option<Path>,
    #[darling(default)]
    pub shader_defs: Option<Path>,
}

#[derive(FromField, Clone)]
#[darling(attributes(shader))]
struct DeriveShadersFieldArgs {
    #[darling(default)]
    pub kernel: Option<String>,
}

#[proc_macro_derive(Shader, attributes(shader))]
pub fn derive_shader(item: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(item as syn::DeriveInput);
    let struct_identifier = &input.ident;

    let derive_shaders = match DeriveShadersArgs::from_derive_input(&input) {
        Ok(v) => v,
        Err(e) => {
            return e.write_errors().into();
        }
    };

    match &input.data {
        Data::Struct(DataStruct { fields, .. }) => {
            /*
             * Field attributes.
             */
            let mut kernels_to_build = vec![];
            let src_path = derive_shaders.src;

            for field in fields.iter() {
                let field_args = match DeriveShadersFieldArgs::from_field(field) {
                    Ok(v) => v,
                    Err(e) => {
                        return e.write_errors().into();
                    }
                };
                let ident = field.ident.as_ref().expect("unnamed fields not supported").into_token_stream();
                let kernel_name = field_args.kernel.map(|k| quote! { #k }).unwrap_or_else(|| quote! { stringify!(#ident) });

                if fields.len() == 1 {
                    // Don’t clone the module if there is only one field.
                    kernels_to_build.push(quote! {
                        #ident: wgcore::utils::load_module(device, #kernel_name, module),
                    });
                } else {
                    kernels_to_build.push(quote! {
                        #ident: wgcore::utils::load_module(device, #kernel_name, module.clone()),
                    });
                }
            }

            let shader_defs = derive_shaders.shader_defs.map(|defs| quote! { #defs() })
                .unwrap_or_else(|| quote! { Default::default() });
            let src = derive_shaders.src_fn.map(|f| quote! { #f(include_str!(#src_path)) })
                .unwrap_or_else(|| quote! { include_str!(#src_path).to_string() });

            let from_device = if !kernels_to_build.is_empty() {
                quote! {
                    let module = Self::composer()
                        .make_naga_module(wgcore::re_exports::naga_oil::compose::NagaModuleDescriptor {
                            source: &Self::src(),
                            file_path: Self::FILE_PATH,
                            shader_defs: #shader_defs,
                            ..Default::default()
                        })
                        .unwrap();
                    Self {
                        #(
                            #kernels_to_build
                        )*
                    }
                }
            } else {
                quote ! {
                    Self
                }
            };

            /*
             * Derive shaders.
             */
            let to_derive: Vec<_> = derive_shaders
                .derive
                .iter()
                .map(|p| p.into_token_stream())
                .collect();
            let composable = derive_shaders.composable.unwrap_or(true);
            quote! {
                #[automatically_derived]
                impl wgcore::shader::Shader for #struct_identifier {
                    const FILE_PATH: &'static str = #src_path;

                    fn from_device(device: &wgcore::re_exports::Device) -> Self {
                        #from_device
                    }

                    // TODO: could we avoid the String allocation here?
                    fn src() -> String {
                        #src
                    }

                    fn compose(composer: &mut wgcore::re_exports::Composer) -> &mut wgcore::re_exports::Composer {
                        use wgcore::composer::ComposerExt;
                        #(
                            #to_derive::compose(composer);
                        )*

                        if #composable {
                            composer
                                .add_composable_module_once(wgcore::re_exports::ComposableModuleDescriptor {
                                    source: &Self::src(),
                                    file_path: Self::FILE_PATH,
                                    shader_defs: #shader_defs,
                                    ..Default::default()
                                })
                                .unwrap();
                        }
                        composer
                    }
                }
            }
        }
        _ => unimplemented!(),
    }
    .into()
}
