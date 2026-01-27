use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(Asset)]
pub fn derive_asset(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = input.ident;

    /*if let syn::Data::Struct(data) = input.data {
        if let syn::Fields::Named(fields) = data.fields {

        }
    }*/

    quote! {
        impl asset_system::assets::Asset for #name {
            fn asset_metadata(&self) -> &AssetMetadata {
                &self.asset_metadata
            }
        }

        impl asset_system::resource_management::Resource for #name {
            fn set_uuid(&mut self, uuid: usize) {
                self.asset_metadata.uuid = uuid;
            }
        }
    }.into()
}
