import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:54:00.791817
# Source Brief: brief_01282.md
# Brief Index: 1282
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls two robotic arms to synchronously
    grab matching, moving objects. The goal is to grab 15 pairs before a timer for
    any single pair runs out.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Control two robotic arms to synchronously grab matching, moving objects. "
        "Score points by successfully grabbing pairs before the timer runs out."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the left arm. Hold SHIFT and use arrow keys to move the "
        "right arm. Press SPACE to attempt a grab with both arms."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Critical Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Screen and Grid Dimensions ---
        self.SCREEN_W, self.SCREEN_H = 640, 400
        self.CELL_SIZE = 40
        self.GRID_W, self.GRID_H = self.SCREEN_W // self.CELL_SIZE, self.SCREEN_H // self.CELL_SIZE

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_timer = pygame.font.SysFont("monospace", 32, bold=True)
        
        # --- Colors ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_LEFT = (0, 255, 150)
        self.COLOR_RIGHT = (0, 150, 255)
        self.COLOR_LEFT_GLOW = (0, 255, 150, 50)
        self.COLOR_RIGHT_GLOW = (0, 150, 255, 50)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TIMER_NORMAL = (0, 255, 150)
        self.COLOR_TIMER_WARN = (255, 50, 50)
        self.COLOR_SUCCESS = (100, 255, 100)
        self.COLOR_FAIL = (255, 100, 100)
        
        # --- Game Constants ---
        self.ARM_SPEED = 1 # grid cells per action
        self.MAX_SCORE = 15
        self.MAX_STEPS = 5000
        self.INITIAL_GRAB_TIME = 5.0
        self.INITIAL_OBJECT_MOVE_DURATION = 2.0
        self.DIFFICULTY_INTERVAL = 3 # Increase difficulty every 3 scores
        self.DIFFICULTY_SPEED_INCREASE = 0.1 # seconds
        self.MIN_OBJECT_MOVE_DURATION = 0.75
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.left_arm_pos = np.array([0, 0])
        self.right_arm_pos = np.array([0, 0])
        
        self.left_obj_pos = np.array([0.0, 0.0])
        self.right_obj_pos = np.array([0.0, 0.0])
        self.left_obj_start_pos = np.array([0.0, 0.0])
        self.right_obj_start_pos = np.array([0.0, 0.0])
        self.left_obj_target_pos = np.array([0, 0])
        self.right_obj_target_pos = np.array([0, 0])
        
        self.object_move_duration = self.INITIAL_OBJECT_MOVE_DURATION
        self.object_move_timer = 0.0
        self.grab_attempt_timer = self.INITIAL_GRAB_TIME
        
        self.prev_space_held = False
        self.particles = []
        self.grab_flash = {"color": None, "timer": 0, "pos": (0, 0)}

        self.left_arm_base = self._grid_to_pixel((self.GRID_W // 4, self.GRID_H))
        self.right_arm_base = self._grid_to_pixel((3 * self.GRID_W // 4, self.GRID_H))
        
        # self.reset() is called by the wrapper
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.particles.clear()
        
        self.object_move_duration = self.INITIAL_OBJECT_MOVE_DURATION
        self.grab_attempt_timer = self.INITIAL_GRAB_TIME
        
        self.left_arm_pos = np.array([self.GRID_W // 4, self.GRID_H - 2])
        self.right_arm_pos = np.array([3 * self.GRID_W // 4, self.GRID_H - 2])
        
        self._spawn_new_objects()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        reward = 0.0
        terminated = self.game_over

        if not terminated:
            dt = 1 / self.metadata["render_fps"]
            
            # --- Update State ---
            self._update_timers(dt)
            self._update_arm_positions(movement, shift_held)
            self._update_object_positions(dt)
            self._update_particles(dt)
            if self.grab_flash['timer'] > 0:
                self.grab_flash['timer'] -= dt

            # --- Handle Grab Action ---
            if space_pressed:
                # sound_grab_attempt.play()
                left_on_target = np.array_equal(self.left_arm_pos, self.left_obj_target_pos)
                right_on_target = np.array_equal(self.right_arm_pos, self.right_obj_target_pos)

                if left_on_target and right_on_target:
                    # sound_grab_success.play()
                    reward += 10.0
                    self.score += 1
                    self.grab_attempt_timer = self.INITIAL_GRAB_TIME
                    self._spawn_new_objects()
                    self._spawn_particles(self.left_arm_pos, self.COLOR_LEFT)
                    self._spawn_particles(self.right_arm_pos, self.COLOR_RIGHT)
                    self._trigger_flash(self.COLOR_SUCCESS, self.left_arm_pos)
                    self._trigger_flash(self.COLOR_SUCCESS, self.right_arm_pos, is_right=True)

                    if self.score > 0 and self.score % self.DIFFICULTY_INTERVAL == 0:
                        self.object_move_duration = max(self.MIN_OBJECT_MOVE_DURATION, self.object_move_duration - self.DIFFICULTY_SPEED_INCREASE)
                else:
                    # sound_grab_fail.play()
                    reward -= 0.1 # Small penalty for a missed attempt
                    self._trigger_flash(self.COLOR_FAIL, self.left_arm_pos)
                    self._trigger_flash(self.COLOR_FAIL, self.right_arm_pos, is_right=True)

            # --- Continuous Reward for Aiming ---
            if np.array_equal(self.left_arm_pos, self.left_obj_target_pos):
                reward += 0.01
            if np.array_equal(self.right_arm_pos, self.right_obj_target_pos):
                reward += 0.01

            # --- Check Termination Conditions ---
            if self.grab_attempt_timer <= 0:
                # sound_game_over.play()
                terminated = True
                reward -= 5.0
            if self.score >= self.MAX_SCORE:
                # sound_win.play()
                terminated = True
                reward += 100.0
            if self.steps >= self.MAX_STEPS:
                terminated = True
        
        self.game_over = terminated
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    # --- State Update Helpers ---

    def _update_timers(self, dt):
        self.grab_attempt_timer -= dt
        self.object_move_timer += dt

    def _update_arm_positions(self, movement, shift_held):
        target_arm_pos = self.right_arm_pos if shift_held else self.left_arm_pos
        
        if movement == 1: target_arm_pos[1] -= self.ARM_SPEED  # Up
        elif movement == 2: target_arm_pos[1] += self.ARM_SPEED  # Down
        elif movement == 3: target_arm_pos[0] -= self.ARM_SPEED  # Left
        elif movement == 4: target_arm_pos[0] += self.ARM_SPEED  # Right
        
        target_arm_pos[0] = np.clip(target_arm_pos[0], 0, self.GRID_W - 1)
        target_arm_pos[1] = np.clip(target_arm_pos[1], 0, self.GRID_H - 1)

    def _update_object_positions(self, dt):
        if self.object_move_timer >= self.object_move_duration:
            self._spawn_new_objects()
        
        t = min(1.0, self.object_move_timer / self.object_move_duration)
        t = 3*t**2 - 2*t**3 # Smoothstep interpolation
        self.left_obj_pos = self._lerp(self.left_obj_start_pos, self.left_obj_target_pos, t)
        self.right_obj_pos = self._lerp(self.right_obj_start_pos, self.right_obj_target_pos, t)

    def _spawn_new_objects(self):
        self.object_move_timer = 0.0
        
        # Set start positions for interpolation
        self.left_obj_start_pos = self.left_obj_target_pos.astype(float) if self.steps > 0 else np.array([self.GRID_W // 4, self.GRID_H // 2], dtype=float)
        self.right_obj_start_pos = self.right_obj_target_pos.astype(float) if self.steps > 0 else np.array([3 * self.GRID_W // 4, self.GRID_H // 2], dtype=float)

        # Generate new, non-overlapping target positions
        while True:
            self.left_obj_target_pos = self.np_random.integers([0, 0], [self.GRID_W, self.GRID_H // 2 + 1])
            self.right_obj_target_pos = self.np_random.integers([0, 0], [self.GRID_W, self.GRID_H // 2 + 1])
            if not np.array_equal(self.left_obj_target_pos, self.right_obj_target_pos):
                break
    
    # --- Rendering and Observation ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid()
        self._draw_objects_and_targets()
        self._draw_arms()
        self._draw_particles()
        self._draw_grab_flash()

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}/{self.MAX_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        timer_val = max(0, self.grab_attempt_timer)
        timer_color = self.COLOR_TIMER_NORMAL if timer_val > 2.0 else self.COLOR_TIMER_WARN
        timer_text = self.font_timer.render(f"{timer_val:.1f}", True, timer_color)
        self.screen.blit(timer_text, (self.SCREEN_W - timer_text.get_width() - 10, 5))

    def _draw_grid(self):
        for x in range(0, self.SCREEN_W, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_H))
        for y in range(0, self.SCREEN_H, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_W, y))

    def _draw_objects_and_targets(self):
        # Draw target zones first
        self._draw_target_zone(self.left_obj_target_pos, self.COLOR_LEFT)
        self._draw_target_zone(self.right_obj_target_pos, self.COLOR_RIGHT)

        # Draw moving objects
        self._draw_glowing_square(self.left_obj_pos, self.COLOR_LEFT, self.COLOR_LEFT_GLOW)
        self._draw_glowing_square(self.right_obj_pos, self.COLOR_RIGHT, self.COLOR_RIGHT_GLOW)

    def _draw_arms(self):
        # Left Arm
        left_px = self._grid_to_pixel(self.left_arm_pos)
        pygame.draw.line(self.screen, self.COLOR_LEFT, self.left_arm_base, left_px, 3)
        self._draw_glowing_circle(left_px, self.COLOR_LEFT, self.COLOR_LEFT_GLOW, 12)
        
        # Right Arm
        right_px = self._grid_to_pixel(self.right_arm_pos)
        pygame.draw.line(self.screen, self.COLOR_RIGHT, self.right_arm_base, right_px, 3)
        self._draw_glowing_circle(right_px, self.COLOR_RIGHT, self.COLOR_RIGHT_GLOW, 12)

    def _draw_grab_flash(self):
        if self.grab_flash['timer'] > 0:
            alpha = int(255 * (self.grab_flash['timer'] / 0.2)) # Fade out over 0.2s
            color = self.grab_flash['color']
            pos = self.grab_flash['pos']
            
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(s, (*color, alpha), s.get_rect())
            self.screen.blit(s, (pos[0] - self.CELL_SIZE // 2, pos[1] - self.CELL_SIZE // 2))

    # --- Visual Effects ---
    
    def _draw_glowing_circle(self, center_px, color, glow_color, radius):
        pygame.gfxdraw.filled_circle(self.screen, center_px[0], center_px[1], radius + 4, glow_color)
        pygame.gfxdraw.filled_circle(self.screen, center_px[0], center_px[1], radius, color)
        pygame.gfxdraw.aacircle(self.screen, center_px[0], center_px[1], radius, color)
        
    def _draw_glowing_square(self, grid_pos, color, glow_color):
        px_pos = self._grid_to_pixel(grid_pos)
        size = int(self.CELL_SIZE * 0.8)
        glow_size = int(self.CELL_SIZE * 1.2)
        
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, glow_color, glow_surf.get_rect(), border_radius=5)
        self.screen.blit(glow_surf, (px_pos[0] - glow_size//2, px_pos[1] - glow_size//2))

        pygame.draw.rect(self.screen, color, (px_pos[0] - size//2, px_pos[1] - size//2, size, size), border_radius=3)

    def _draw_target_zone(self, grid_pos, color):
        px_pos = self._grid_to_pixel(grid_pos)
        rect = pygame.Rect(px_pos[0] - self.CELL_SIZE//2, px_pos[1] - self.CELL_SIZE//2, self.CELL_SIZE, self.CELL_SIZE)
        
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, (*color, 20), s.get_rect())
        pygame.draw.rect(s, (*color, 60), s.get_rect(), 2)
        self.screen.blit(s, rect.topleft)

    def _spawn_particles(self, grid_pos, color):
        px_pos = self._grid_to_pixel(grid_pos)
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(50, 150)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = random.uniform(0.3, 0.8)
            self.particles.append({'pos': list(px_pos), 'vel': vel, 'life': lifespan, 'max_life': lifespan, 'color': color})

    def _update_particles(self, dt):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0] * dt
            p['pos'][1] += p['vel'][1] * dt
            p['vel'][1] += 150 * dt # Gravity
            p['life'] -= dt
            if p['life'] <= 0:
                self.particles.remove(p)

    def _draw_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            radius = int(life_ratio * 4)
            if radius > 0:
                color_with_alpha = (*p['color'], int(life_ratio * 255))
                s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color_with_alpha, (radius, radius), radius)
                self.screen.blit(s, (int(p['pos'][0]-radius), int(p['pos'][1]-radius)))

    def _trigger_flash(self, color, grid_pos, is_right=False):
        # Flashes are independent now, so we need two flash states
        # For simplicity in this implementation, we just overwrite.
        # A more complex implementation would use a list of flashes.
        self.grab_flash = {
            "color": color,
            "timer": 0.2,
            "pos": self._grid_to_pixel(grid_pos)
        }

    # --- Utilities ---

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _grid_to_pixel(self, grid_pos):
        px = (grid_pos[0] + 0.5) * self.CELL_SIZE
        py = (grid_pos[1] + 0.5) * self.CELL_SIZE
        return int(px), int(py)

    def _lerp(self, a, b, t):
        return a * (1.0 - t) + b * t

    def close(self):
        pygame.quit()

# --- Example Usage ---
if __name__ == "__main__":
    # The original __main__ block was using pygame.display which is not compatible with
    # the headless mode (SDL_VIDEODRIVER="dummy"). It has been removed to ensure
    # the script is runnable in a server environment.
    # A simple test of the environment API is provided instead.
    
    print("Creating and testing the environment...")
    env = GameEnv()
    
    # Test reset
    obs, info = env.reset()
    print(f"Reset successful. Observation shape: {obs.shape}, Info: {info}")
    
    # Test a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
        if terminated or truncated:
            print("Episode finished. Resetting.")
            env.reset()
            
    env.close()
    print("Environment test complete.")