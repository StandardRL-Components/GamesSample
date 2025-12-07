
# Generated: 2025-08-28T00:21:51.938252
# Source Brief: brief_03768.md
# Brief Index: 3768

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to set rotation. ↑ for 90°, ↓ for 180°, ← for 270°, → for 0°."
    )

    game_description = (
        "A fast-paced puzzle game. Rotate falling shapes to match the target slots below before they land. Complete three levels of increasing difficulty to win."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 5000
    MAX_ATTEMPTS = 5
    NUM_LEVELS = 3
    FIT_Y_LEVEL = 350
    BASE_FALL_SPEED = 2.0
    
    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_SLOT_OUTLINE = (60, 70, 90)
    COLOR_FAILURE_FLASH = (180, 50, 50, 100)
    
    SHAPE_COLORS = [
        (50, 200, 255),  # Cyan - Square
        (255, 200, 50),  # Yellow - Triangle
        (200, 50, 255),  # Magenta - L-Shape
        (50, 255, 150),  # Green - T-Shape
    ]

    # --- Shape Definitions (vertices relative to origin) ---
    SHAPE_VERTICES = {
        0: [(-20, -20), (20, -20), (20, 20), (-20, 20)],  # Square
        1: [(-25, 20), (25, 20), (0, -25)],  # Triangle
        2: [(-25, -25), (25, -25), (25, -5), (5, -5), (5, 25), (-25, 25)],  # L-Shape
        3: [(-25, -25), (25, -25), (25, -5), (5, -5), (5, 25), (-5, 25), (-5, -5), (-25, -5)], # T-Shape
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_level = pygame.font.Font(None, 48)
        self.font_msg = pygame.font.Font(None, 64)

        self.current_shape = None
        self.slots = []
        self.shape_queue = []
        self.particles = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.attempts_remaining = self.MAX_ATTEMPTS
        
        self.particles.clear()
        self.flash_timer = 0
        self.level_transition_timer = 0
        self.next_shape_timer = 60 # Initial delay
        
        self._setup_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        
        self._handle_action(action)
        reward += self._update_game_state()
        
        terminated = self.game_over or (self.steps >= self.MAX_STEPS)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        if self.current_shape is None:
            return

        movement = action[0]
        # Absolute rotation setting as per brief
        if movement == 1: self.current_shape['rotation'] = 90
        elif movement == 2: self.current_shape['rotation'] = 180
        elif movement == 3: self.current_shape['rotation'] = 270
        elif movement == 4: self.current_shape['rotation'] = 0
        # movement == 0 is no-op

    def _update_game_state(self):
        step_reward = 0

        # Update timers
        if self.flash_timer > 0: self.flash_timer -= 1
        if self.level_transition_timer > 0:
            self.level_transition_timer -= 1
            if self.level_transition_timer == 0:
                self._setup_level()
                self.next_shape_timer = 30
            return step_reward

        # Update particles
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles: p.update()

        # Handle shape spawning
        if self.current_shape is None:
            if self.next_shape_timer > 0:
                self.next_shape_timer -= 1
            else:
                self._spawn_shape()
            return step_reward

        # Update current shape
        self.current_shape['pos'][1] += self.fall_speed
        step_reward += 0.01 # Small reward for staying in the game

        # Check for landing
        if self.current_shape['pos'][1] >= self.FIT_Y_LEVEL:
            is_fit, target_slot = self._check_fit()
            
            if is_fit:
                # --- Success ---
                # sfx: success_chime.wav
                step_reward += 10
                self.score += 10
                target_slot['filled'] = True
                self._spawn_particles(target_slot['pos'], target_slot['color'])
                self.level_shapes_cleared += 1

                if self.level_shapes_cleared == len(self.slots):
                    # --- Level Complete ---
                    # sfx: level_complete.wav
                    step_reward += 50
                    self.score += 50
                    self.level += 1
                    if self.level > self.NUM_LEVELS:
                        # --- Game Won ---
                        # sfx: victory_fanfare.wav
                        step_reward += 100
                        self.score += 100
                        self.game_over = True
                    else:
                        self.level_transition_timer = 90 # Frames for "Level Cleared" message
            else:
                # --- Failure ---
                # sfx: failure_buzz.wav
                step_reward -= 5
                self.attempts_remaining -= 1
                self.flash_timer = 15 # Frames for red flash
                if self.attempts_remaining <= 0:
                    step_reward -= 100
                    self.game_over = True
            
            self.current_shape = None
            self.next_shape_timer = 20 # Delay before next shape

        return step_reward

    def _check_fit(self):
        shape = self.current_shape
        # Find the corresponding unfilled slot for this shape type
        target_slot = next((s for s in self.slots if s['type'] == shape['type'] and not s['filled']), None)
        
        if target_slot is None:
            return False, None

        # Check position alignment
        pos_match = abs(shape['pos'][0] - target_slot['pos'][0]) < 5

        # Check rotation alignment
        # Squares can fit at any 90-degree rotation
        if shape['type'] == 0: # Square
            rot_match = True
        else:
            rot_match = shape['rotation'] == target_slot['target_rotation']

        if pos_match and rot_match:
            return True, target_slot
        
        return False, None

    def _setup_level(self):
        self.slots.clear()
        self.shape_queue.clear()
        self.fall_speed = self.BASE_FALL_SPEED * (1.1 ** (self.level - 1))
        self.level_shapes_cleared = 0

        if self.level == 1:
            shapes_in_level = [(0, 0), (1, 180), (2, 90)] # type, rotation
            positions = [160, 320, 480]
        elif self.level == 2:
            shapes_in_level = [(1, 0), (3, 270), (0, 0), (2, 180)]
            positions = [120, 260, 400, 540]
        elif self.level == 3:
            shapes_in_level = [(2, 0), (3, 90), (1, 270), (0, 0), (3, 180)]
            positions = [100, 210, 320, 430, 540]
        else:
            return # Game is won/over

        for i, (shape_type, rotation) in enumerate(shapes_in_level):
            pos_x = positions[i]
            color = self.SHAPE_COLORS[shape_type]
            self.slots.append({
                'pos': [pos_x, self.FIT_Y_LEVEL],
                'target_rotation': rotation,
                'type': shape_type,
                'color': color,
                'filled': False
            })
            self.shape_queue.append({'type': shape_type, 'color': color})
        
        self.np_random.shuffle(self.shape_queue)

    def _spawn_shape(self):
        if not self.shape_queue:
            self.current_shape = None
            return
        
        shape_info = self.shape_queue.pop(0)
        self.current_shape = {
            'type': shape_info['type'],
            'color': shape_info['color'],
            'pos': [self.SCREEN_WIDTH // 2, -30],
            'rotation': 0,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render slots
        for slot in self.slots:
            points = self._rotate_points(self.SHAPE_VERTICES[slot['type']], slot['target_rotation'])
            abs_points = [(p[0] + slot['pos'][0], p[1] + slot['pos'][1]) for p in points]
            
            if slot['filled']:
                # Bright, filled effect
                color = tuple(min(255, c + 80) for c in slot['color'])
                pygame.gfxdraw.filled_polygon(self.screen, abs_points, color)
                pygame.gfxdraw.aapolygon(self.screen, abs_points, color)
            else:
                pygame.gfxdraw.aapolygon(self.screen, abs_points, self.COLOR_SLOT_OUTLINE)

        # Render current shape
        if self.current_shape:
            shape = self.current_shape
            points = self._rotate_points(self.SHAPE_VERTICES[shape['type']], shape['rotation'])
            abs_points = [(p[0] + shape['pos'][0], p[1] + shape['pos'][1]) for p in points]
            pygame.gfxdraw.filled_polygon(self.screen, abs_points, shape['color'])
            pygame.gfxdraw.aapolygon(self.screen, abs_points, tuple(min(255, c+50) for c in shape['color']))
        
        # Render particles
        for p in self.particles: p.draw(self.screen)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Attempts
        attempts_text = self.font_main.render(f"Attempts: {self.attempts_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(attempts_text, (self.SCREEN_WIDTH - attempts_text.get_width() - 10, 10))
        
        # Level
        level_text = self.font_level.render(f"Level {self.level}", True, self.COLOR_UI_TEXT)
        level_rect = level_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 25))
        self.screen.blit(level_text, level_rect)

        # Failure flash
        if self.flash_timer > 0:
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = int(100 * (self.flash_timer / 15))
            flash_surface.fill((*self.COLOR_FAILURE_FLASH[:3], alpha))
            self.screen.blit(flash_surface, (0, 0))

        # Level transition message
        if self.level_transition_timer > 0 and not self.game_over:
            msg = "Level Cleared!"
            msg_text = self.font_msg.render(msg, True, self.COLOR_UI_TEXT)
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(msg_text, msg_rect)

        # Game over / Victory message
        if self.game_over:
            msg = "VICTORY!" if self.attempts_remaining > 0 else "GAME OVER"
            color = (100, 255, 100) if self.attempts_remaining > 0 else (255, 100, 100)
            msg_text = self.font_msg.render(msg, True, color)
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _rotate_points(self, points, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        return [
            (p[0] * cos_a - p[1] * sin_a, p[0] * sin_a + p[1] * cos_a) for p in points
        ]

    def _spawn_particles(self, pos, color):
        # sfx: particle_burst.wav
        for _ in range(30):
            self.particles.append(Particle(pos[0], pos[1], color, self.np_random))

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

class Particle:
    def __init__(self, x, y, color, rng):
        self.x = x
        self.y = y
        self.color = color
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(1, 5)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifespan = rng.integers(25, 45)
        self.initial_lifespan = self.lifespan

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.98
        self.vy *= 0.98
        self.lifespan -= 1

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = max(0, min(255, int(255 * (self.lifespan / self.initial_lifespan))))
            size = int(3 * (self.lifespan / self.initial_lifespan))
            if size > 0:
                pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), size, (*self.color, alpha))

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("GeoFit")
    
    terminated = False
    total_reward = 0
    
    # Mapping keyboard keys to actions for manual play
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    running = True
    while running:
        action = np.array([0, 0, 0]) # Default no-op action
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    action[0] = key_to_action[event.key]
                if event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    total_reward = 0
                    terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30) # Run at 30 FPS

    pygame.quit()