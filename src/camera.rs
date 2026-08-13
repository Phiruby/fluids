use crate::Camera;
use winit::event::{ElementState, MouseButton};
use winit::keyboard::KeyCode;

pub struct CameraController {
    speed: f32,
    pan_sensitivity: f32,
    is_right_mouse_down: bool,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_strafe_left_pressed: bool,
    is_strafe_right_pressed: bool,
    last_cursor: Option<cgmath::Point2<f64>>,
}

impl CameraController {
    pub fn new() -> Self {
        Self {
            speed: 0.1,
            pan_sensitivity: 0.02,
            is_right_mouse_down: false,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_strafe_left_pressed: false,
            is_strafe_right_pressed: false,
            last_cursor: None,
        }
    }

    pub fn process_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        if button != MouseButton::Right {
            return;
        }
        self.is_right_mouse_down = state == ElementState::Pressed;
        if !self.is_right_mouse_down {
            self.last_cursor = None;
        }
    }

    pub fn process_cursor_moved(
        &mut self,
        position: winit::dpi::PhysicalPosition<f64>,
        camera: &mut Camera,
    ) {
        let current = cgmath::Point2::new(position.x, position.y);
        if self.is_right_mouse_down {
            if let Some(prev) = self.last_cursor {
                let dx = (current.x - prev.x) as f32;
                let dy = (current.y - prev.y) as f32;
                self.pan(camera, dx, dy);
            }
        }
        self.last_cursor = Some(current);
    }

    pub fn process_key(&mut self, code: KeyCode, is_pressed: bool) -> bool {
        match code {
            KeyCode::KeyW | KeyCode::ArrowUp => {
                self.is_forward_pressed = is_pressed;
                true
            }
            KeyCode::KeyA | KeyCode::ArrowLeft => {
                self.is_strafe_left_pressed = is_pressed;
                true
            }
            KeyCode::KeyS | KeyCode::ArrowDown => {
                self.is_backward_pressed = is_pressed;
                true
            }
            KeyCode::KeyD | KeyCode::ArrowRight => {
                self.is_strafe_right_pressed = is_pressed;
                true
            }
            _ => false,
        }
    }

    pub fn update_camera(&self, camera: &mut Camera) {
        use cgmath::InnerSpace;
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        let forward_mag = forward.magnitude();

        if self.is_forward_pressed && forward_mag > self.speed {
            camera.eye += forward_norm * self.speed;
        }
        if self.is_backward_pressed {
            camera.eye -= forward_norm * self.speed;
        }

        let right = forward_norm.cross(camera.up);

        let forward = camera.target - camera.eye;
        let forward_mag = forward.magnitude();

        if self.is_strafe_right_pressed {
            camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
        }
        if self.is_strafe_left_pressed {
            camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
        }
    }

    fn pan(&self, camera: &mut Camera, dx: f32, dy: f32) {
        camera.target +=
            cgmath::Vector3::new(dx * self.pan_sensitivity, -dy * self.pan_sensitivity, 0.0);
    }
}
