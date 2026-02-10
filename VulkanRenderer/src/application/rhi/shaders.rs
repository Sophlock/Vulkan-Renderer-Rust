use std::path::Path;

use shader_slang::{
    CompileTarget, CompilerOptions, GlobalSession, OptimizationLevel, Session, SessionDesc,
    TargetDesc,
};

pub struct SlangCompiler {
    session: Session,
}

impl SlangCompiler {
    pub fn new(shader_base_path: &Path) -> Self {
        let global_session = GlobalSession::new().unwrap();
        println!(
            "Using slang compiler version {}",
            global_session.build_tag_string()
        );
        let targets = [TargetDesc::default()
            .format(CompileTarget::Spirv)
            .profile(global_session.find_profile("spirv_1_6"))];
        let search_paths = [shader_base_path.to_str().unwrap().as_ptr() as *const i8];
        let options = CompilerOptions::default()
            .optimization(OptimizationLevel::High)
            // TODO: Once Vulkano supports the vulkan extension for compute shader derivatives (or slang doesn't use it anymore), we should directly emit spirv code
            .emit_spirv_directly(false)
            .matrix_layout_column(true);
        let session_description = SessionDesc::default()
            .targets(&targets)
            .search_paths(search_paths.as_slice())
            .options(&options);
        let session = global_session.create_session(&session_description).unwrap();
        Self { session }
    }

    pub fn session(&self) -> &Session {
        &self.session
    }
}
