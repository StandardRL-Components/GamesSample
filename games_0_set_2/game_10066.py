import gymnasium as gym
import os
import pygame
import numpy as np
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:51:26.639203
# Source Brief: brief_00066.md
# Brief Index: 66
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw

# Helper class for particle effects
class Particle:
    def __init__(self, x, y, color, np_random):
        self.x = x
        self.y = y
        self.color = color
        self.np_random = np_random
        self.vx = self.np_random.uniform(-3, 3)
        self.vy = self.np_random.uniform(-5, -1)
        self.lifespan = self.np_random.uniform(20, 40)
        self.alpha = 255

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.2  # Gravity
        self.lifespan -= 1
        self.alpha = max(0, self.alpha - 255 / self.lifespan if self.lifespan > 0 else 255)

    def draw(self, surface):
        if self.lifespan > 0:
            s = pygame.Surface((6, 6), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.color, self.alpha), (3, 3), 3)
            surface.blit(s, (int(self.x) - 3, int(self.y) - 3))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "Rotate a cube to match its color with the briefly flashing target color at the top of the screen. "
        "Match colors correctly to score points before time runs out or you make too many mistakes."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to change the color of the cube. Match the target color before the next check."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.MAX_STEPS = self.FPS * 60  # 60 seconds
        self.MAX_MISMATCHES = 5
        self.CHECK_INTERVAL = self.FPS // 2  # 0.5 seconds
        self.NEW_COLOR_INTERVAL = self.FPS * 15 # 15 seconds

        # --- Colors ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_SUCCESS = (100, 255, 100)
        self.COLOR_FAIL = (255, 100, 100)
        
        self.FULL_COLOR_PALETTE = [
            (255, 60, 60),   # Red
            (60, 60, 255),   # Blue
            (60, 255, 60),   # Green
            (255, 255, 60),  # Yellow
            (255, 165, 0),   # Orange
            (128, 0, 128),   # Purple
            (0, 255, 255),   # Cyan
            (255, 0, 255),   # Magenta
        ]
        self.INITIAL_COLOR_COUNT = 4

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 24)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.mismatches = 0
        self.time_since_last_check = 0
        self.available_colors = []
        self.current_color_idx = 0
        self.target_color_idx = 0
        self.flash_color_idx = 0
        self.last_reward = 0
        
        # --- Visuals State ---
        self.cube_rotation_angle = 0.0
        self.target_cube_rotation_angle = 0.0
        self.particles = []
        self.feedback_alpha = 0
        self.last_match_result = 'success'

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.mismatches = 0
        self.time_since_last_check = 0
        
        self.available_colors = self.FULL_COLOR_PALETTE[:self.INITIAL_COLOR_COUNT]
        self.current_color_idx = 0
        self.target_color_idx = self.np_random.integers(0, len(self.available_colors))
        self.flash_color_idx = self.target_color_idx
        
        self.cube_rotation_angle = 0.0
        self.target_cube_rotation_angle = 0.0
        self.particles = []
        self.feedback_alpha = 0
        self.last_reward = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self._handle_input(movement)
        
        self.steps += 1
        self.time_since_last_check += 1
        
        reward = self._update_game_state()
        
        terminated = (self.mismatches >= self.MAX_MISMATCHES) or (self.steps >= self.MAX_STEPS)
        if terminated and not self.game_over:
            self.game_over = True
            if self.mismatches >= self.MAX_MISMATCHES:
                reward -= 100 # Penalty for failing
            else:
                reward += 100 # Bonus for surviving
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 0: # No-op
            return
            
        num_colors = len(self.available_colors)
        if num_colors == 0: return

        prev_idx = self.current_color_idx
        if movement in [1, 4]: # Up or Right -> Next color
            self.current_color_idx = (self.current_color_idx + 1) % num_colors
        elif movement in [2, 3]: # Down or Left -> Previous color
            self.current_color_idx = (self.current_color_idx - 1 + num_colors) % num_colors

        if self.current_color_idx != prev_idx:
            # Add rotation for visual feedback
            self.target_cube_rotation_angle += 90 * (1 if movement in [1, 4] else -1)
            # sfx: cube_rotate.wav

    def _update_game_state(self):
        reward = 0
        
        # Add new colors periodically
        num_new_colors_to_add = (self.steps // self.NEW_COLOR_INTERVAL)
        target_color_count = min(len(self.FULL_COLOR_PALETTE), self.INITIAL_COLOR_COUNT + num_new_colors_to_add)
        if len(self.available_colors) < target_color_count:
            self.available_colors = self.FULL_COLOR_PALETTE[:target_color_count]
            # sfx: new_color_unlocked.wav

        # Perform color check at intervals
        if self.time_since_last_check >= self.CHECK_INTERVAL:
            self.time_since_last_check = 0
            
            # Check for match
            if self.current_color_idx == self.target_color_idx:
                reward = 1.0
                self.score += 1
                self.feedback_alpha = 255
                self.last_match_result = 'success'
                # sfx: match_success.wav
                self._create_particles(self.available_colors[self.target_color_idx])
            else:
                self.mismatches += 1
                self.feedback_alpha = 255
                self.last_match_result = 'fail'
                # sfx: match_fail.wav
                self._create_particles(self.COLOR_FAIL)

            # Select next target color and flash color
            self.target_color_idx = self.np_random.integers(0, len(self.available_colors))
            is_decoy = self.np_random.random() < 0.1 and len(self.available_colors) > 1
            if is_decoy:
                possible_decoys = list(range(len(self.available_colors)))
                possible_decoys.remove(self.target_color_idx)
                self.flash_color_idx = self.np_random.choice(possible_decoys)
            else:
                self.flash_color_idx = self.target_color_idx
        
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Update and draw particles
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.draw(self.screen)

        # Draw the flash indicator
        self._render_flash_indicator()

        # Draw the central cube
        self._render_cube()
        
        # Draw feedback flash
        if self.feedback_alpha > 0:
            color = self.COLOR_SUCCESS if self.last_match_result == 'success' else self.COLOR_FAIL
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((*color, self.feedback_alpha))
            self.screen.blit(flash_surface, (0, 0))
            self.feedback_alpha = max(0, self.feedback_alpha - 15)

    def _render_flash_indicator(self):
        flash_color = self.available_colors[self.flash_color_idx]
        bar_height = 20
        # Create a glow effect
        for i in range(bar_height, 0, -2):
            alpha = 150 * (1 - i / bar_height)
            glow_color = (*flash_color, alpha)
            pygame.gfxdraw.box(self.screen, (0, 0, self.SCREEN_WIDTH, i), glow_color)
        pygame.draw.rect(self.screen, flash_color, (0, 0, self.SCREEN_WIDTH, 5))

    def _render_cube(self):
        # Smoothly interpolate rotation
        self.cube_rotation_angle += (self.target_cube_rotation_angle - self.cube_rotation_angle) * 0.2
        
        cx, cy = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 50
        size = 60
        
        angle_rad = math.radians(self.cube_rotation_angle)
        
        # Define points for an isometric-style cube
        p = {
            'top_left': (cx - size * math.cos(angle_rad), cy - size * math.sin(angle_rad) - size/2),
            'top_right': (cx + size * math.cos(angle_rad), cy + size * math.sin(angle_rad) - size/2),
            'front': (cx, cy),
            'back': (cx, cy - size),
            'bottom_left': (cx - size * math.cos(angle_rad), cy - size * math.sin(angle_rad) + size/2),
            'bottom_right': (cx + size * math.cos(angle_rad), cy + size * math.sin(angle_rad) + size/2),
        }

        # Get current color
        current_color = self.available_colors[self.current_color_idx]
        
        # Darken for side faces
        side_color = tuple(max(0, c - 50) for c in current_color)
        dark_side_color = tuple(max(0, c - 80) for c in side_color)

        # Draw faces
        # Right face
        pygame.draw.polygon(self.screen, side_color, [p['front'], p['top_right'], p['back'], p['bottom_right']])
        pygame.draw.aalines(self.screen, self.COLOR_UI_TEXT, True, [p['front'], p['top_right'], p['back'], p['bottom_right']])
        # Left face
        pygame.draw.polygon(self.screen, dark_side_color, [p['front'], p['top_left'], p['back'], p['bottom_left']])
        pygame.draw.aalines(self.screen, self.COLOR_UI_TEXT, True, [p['front'], p['top_left'], p['back'], p['bottom_left']])
        # Top face (main color)
        pygame.draw.polygon(self.screen, current_color, [p['top_left'], p['top_right'], p['back']])
        pygame.draw.aalines(self.screen, self.COLOR_UI_TEXT, True, [p['top_left'], p['top_right'], p['back']])

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, self.SCREEN_HEIGHT - 50))
        
        # Mismatches
        mismatch_text = self.font_small.render("MISMATCHES:", True, self.COLOR_UI_TEXT)
        self.screen.blit(mismatch_text, (self.SCREEN_WIDTH - 250, self.SCREEN_HEIGHT - 50))
        for i in range(self.MAX_MISMATCHES):
            color = self.COLOR_FAIL if i < self.mismatches else (80, 80, 90)
            pygame.draw.circle(self.screen, color, (self.SCREEN_WIDTH - 110 + i * 25, self.SCREEN_HEIGHT - 38), 8)

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_large.render(f"{time_left:.1f}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH // 2 - timer_text.get_width() // 2, 30))

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_text = "SURVIVED!" if self.mismatches < self.MAX_MISMATCHES else "GAME OVER"
            text_surface = self.font_large.render(win_text, True, self.COLOR_SUCCESS if self.mismatches < self.MAX_MISMATCHES else self.COLOR_FAIL)
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surface, text_rect)

    def _create_particles(self, color):
        cx, cy = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 50
        for _ in range(30):
            self.particles.append(Particle(cx, cy - 30, color, self.np_random))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "mismatches": self.mismatches,
            "colors_in_play": len(self.available_colors)
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Color Cube")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    
    while not done:
        # --- Action Mapping for Manual Play ---
        movement = 0 # No-op
        action = [movement, 0, 0] # Space and Shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key in [pygame.K_UP, pygame.K_RIGHT]:
                    action[0] = 1
                elif event.key in [pygame.K_DOWN, pygame.K_LEFT]:
                    action[0] = 2

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            # Render one last time to show game over screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.time.wait(2000) # Wait 2 seconds
            
            obs, info = env.reset()
            total_reward = 0

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.metadata['render_fps'])
        
    env.close()