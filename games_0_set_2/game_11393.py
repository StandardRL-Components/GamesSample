import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:58:08.026339
# Source Brief: brief_01393.md
# Brief Index: 1393
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "Stack layers of a new world from a moving conveyor belt. "
        "Control the belt's speed and time your drops perfectly to build the planet before time runs out."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to change the belt speed and press space to drop a layer."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.MAX_STEPS = 1800  # 60 seconds at 30 FPS
        self.WIN_CONDITION_LAYERS = 10
        self.TARGET_FPS = 30

        # --- Gameplay Constants ---
        self.MIN_BELT_SPEED = 0.5
        self.MAX_BELT_SPEED = 4.0
        self.SPEED_INCREMENT = 0.1
        self.LAYER_WIDTH, self.LAYER_HEIGHT = 80, 20
        self.STACK_X_POS = self.SCREEN_WIDTH // 2
        self.STACK_WIDTH = 100
        self.DEPOSIT_RANGE = self.STACK_WIDTH * 0.8
        self.SYNC_BONUS_TIME = 90 # 3 seconds at 30fps
        self.NUM_LAYERS_ON_BELT = 5

        # --- Visual Constants ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_BELT = (50, 55, 70)
        self.COLOR_BELT_MARKER = (70, 75, 90)
        self.COLOR_TEXT = (220, 230, 255)
        self.COLOR_TEXT_SHADOW = (10, 15, 25)
        self.COLOR_GREEN = (0, 255, 150)
        self.COLOR_RED = (255, 80, 80)
        self.LAYER_TYPES = [
            {"name": "Water", "color": (100, 150, 255)},
            {"name": "Soil", "color": (180, 140, 90)},
            {"name": "Atmosphere", "color": (180, 220, 255)},
        ]
        self.BELT_Y_POS = 280

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_bonus = pygame.font.Font(None, 28)
        
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.belt_speed = 0.0
        self.belt_marker_offset = 0.0
        self.layers_on_belt = []
        self.stacked_layers = []
        self.particles = []
        self.last_space_held = False
        self.last_stack_time = 0
        self.sync_bonus_flash_timer = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.belt_speed = 1.0
        self.belt_marker_offset = 0.0
        self.layers_on_belt = []
        self.stacked_layers = []
        self.particles = []
        self.last_space_held = False
        self.last_stack_time = -self.SYNC_BONUS_TIME * 2 # Allow bonus on first stack
        self.sync_bonus_flash_timer = 0

        self._spawn_initial_layers()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        
        self._handle_input(action)
        reward += self._handle_stacking(action)
        self._update_game_state()
        
        self.steps += 1
        
        win = len(self.stacked_layers) >= self.WIN_CONDITION_LAYERS
        lose = self.steps >= self.MAX_STEPS
        terminated = win or lose
        
        if terminated and not self.game_over:
            if win:
                reward += 100.0
                # print("VICTORY! Final Score:", self.score + reward)
            else: # lose
                reward -= 100.0
                # print("DEFEAT! Final Score:", self.score + reward)
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        
        if movement == 3: # Left
            self.belt_speed -= self.SPEED_INCREMENT
        elif movement == 4: # Right
            self.belt_speed += self.SPEED_INCREMENT
        
        self.belt_speed = np.clip(self.belt_speed, self.MIN_BELT_SPEED, self.MAX_BELT_SPEED)

    def _handle_stacking(self, action):
        space_held = action[1] == 1
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held

        if not space_pressed:
            return 0.0

        # Find the layer closest to the stacking zone
        target_layer = None
        min_dist = float('inf')
        for layer in self.layers_on_belt:
            dist = abs(layer['x'] + self.LAYER_WIDTH / 2 - self.STACK_X_POS)
            if dist < min_dist and dist < self.DEPOSIT_RANGE:
                min_dist = dist
                target_layer = layer
        
        if target_layer is None:
            return 0.0
        
        # --- Layer is deposited ---
        # Sfx: deposit_attempt.wav
        
        # Calculate alignment (0 to 1, where 1 is perfect)
        alignment = 1.0 - (min_dist / (self.STACK_WIDTH / 2))
        alignment = max(0, alignment)
        
        # Base reward for alignment
        stacking_reward = 0.2 if alignment > 0.95 else 0.1
        
        # Add to stacked layers
        self.stacked_layers.append({
            'type_info': target_layer['type_info'],
            'alignment': alignment,
            'wobble': self.np_random.uniform(-1, 1) * (1 - alignment)
        })
        self.layers_on_belt.remove(target_layer)

        # Sfx: stack_success.wav
        stacking_reward += 1.0

        # Check for sync bonus
        if self.steps - self.last_stack_time < self.SYNC_BONUS_TIME:
            stacking_reward += 2.0
            self.sync_bonus_flash_timer = self.TARGET_FPS // 2 # Flash for 0.5s
            # Sfx: sync_bonus.wav

        self.last_stack_time = self.steps
        
        # Create particles
        stack_pos_y = self.BELT_Y_POS - (len(self.stacked_layers) -1) * (self.LAYER_HEIGHT * 0.75)
        self._create_particles(self.STACK_X_POS, stack_pos_y, target_layer['type_info']['color'])
        
        return stacking_reward

    def _update_game_state(self):
        # Move layers on belt
        for layer in self.layers_on_belt:
            layer['x'] -= self.belt_speed
        
        # Update belt visual
        self.belt_marker_offset = (self.belt_marker_offset - self.belt_speed) % 50

        # Remove off-screen layers and spawn new ones
        self.layers_on_belt = [layer for layer in self.layers_on_belt if layer['x'] > -self.LAYER_WIDTH]
        while len(self.layers_on_belt) < self.NUM_LAYERS_ON_BELT:
            self._spawn_layer()

        # Update particles
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        
        # Update timers
        if self.sync_bonus_flash_timer > 0:
            self.sync_bonus_flash_timer -= 1
            
    def _spawn_initial_layers(self):
        for i in range(self.NUM_LAYERS_ON_BELT):
            x_pos = self.SCREEN_WIDTH + i * (self.SCREEN_WIDTH / self.NUM_LAYERS_ON_BELT)
            self._spawn_layer(x_pos)

    def _spawn_layer(self, x_pos=None):
        if x_pos is None:
            # Find the rightmost layer to spawn after it
            max_x = self.SCREEN_WIDTH
            if self.layers_on_belt:
                max_x = max(layer['x'] for layer in self.layers_on_belt)
            x_pos = max_x + self.np_random.uniform(200, 300)

        layer_type = self.np_random.choice(self.LAYER_TYPES)
        self.layers_on_belt.append({
            'x': x_pos,
            'type_info': layer_type,
        })
        
    def _create_particles(self, x, y, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'x': x,
                'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'lifespan': self.np_random.integers(20, 40),
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_conveyor_belt()
        self._render_planet()
        self._render_layers_on_belt()
        self._render_alignment_indicator()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "layers_stacked": len(self.stacked_layers),
            "belt_speed": self.belt_speed,
        }

    def _render_text(self, text, x, y, font, color, shadow=True, center=False):
        if shadow:
            text_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            text_rect = text_surf.get_rect()
            if center:
                text_rect.center = (x + 2, y + 2)
            else:
                text_rect.topleft = (x + 2, y + 2)
            self.screen.blit(text_surf, text_rect)

        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = (x, y)
        else:
            text_rect.topleft = (x, y)
        self.screen.blit(text_surf, text_rect)

    def _render_background(self):
        for i in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

    def _render_conveyor_belt(self):
        belt_rect = pygame.Rect(0, self.BELT_Y_POS, self.SCREEN_WIDTH, 80)
        pygame.draw.rect(self.screen, self.COLOR_BELT, belt_rect)
        pygame.draw.rect(self.screen, (0,0,0), belt_rect, 2)
        
        for i in range(-50, self.SCREEN_WIDTH + 50, 50):
            x = int(i + self.belt_marker_offset)
            pygame.draw.line(self.screen, self.COLOR_BELT_MARKER, (x, self.BELT_Y_POS), (x+25, self.BELT_Y_POS + 80), 2)

    def _render_planet(self):
        base_y = self.BELT_Y_POS
        if not self.stacked_layers:
            # Draw a base platform
            rect = pygame.Rect(0,0, 60, 10)
            rect.center = (self.STACK_X_POS, base_y + 5)
            pygame.draw.rect(self.screen, (100,100,120), rect, border_radius=3)
        
        for i, layer in enumerate(self.stacked_layers):
            y_pos = base_y - i * (self.LAYER_HEIGHT * 0.75)
            width = self.LAYER_WIDTH + 20 - i * 2
            height = self.LAYER_HEIGHT
            
            x_offset = layer['wobble'] * 10
            
            color = layer['type_info']['color']
            dark_color = (max(0, c-40) for c in color)
            
            # Draw 3D-ish layer
            rect = pygame.Rect(0, 0, width, height)
            rect.center = (self.STACK_X_POS + x_offset, y_pos)
            
            pygame.draw.rect(self.screen, tuple(dark_color), (rect.x, rect.y + 5, rect.width, rect.height), border_radius=8)
            pygame.draw.rect(self.screen, color, rect, border_radius=8)
            pygame.draw.rect(self.screen, (255,255,255,50), rect, 1, border_radius=8)

    def _render_layers_on_belt(self):
        for layer in self.layers_on_belt:
            x, y = int(layer['x']), int(self.BELT_Y_POS - self.LAYER_HEIGHT)
            color = layer['type_info']['color']
            dark_color = (max(0, c-40) for c in color)
            
            # Draw 3D-ish layer
            pygame.draw.rect(self.screen, tuple(dark_color), (x, y + 5, self.LAYER_WIDTH, self.LAYER_HEIGHT))
            pygame.draw.rect(self.screen, color, (x, y, self.LAYER_WIDTH, self.LAYER_HEIGHT))
            pygame.draw.rect(self.screen, (0,0,0,50), (x, y, self.LAYER_WIDTH, self.LAYER_HEIGHT), 1)

    def _render_alignment_indicator(self):
        target_layer = None
        min_dist = float('inf')
        for layer in self.layers_on_belt:
            dist = abs(layer['x'] + self.LAYER_WIDTH / 2 - self.STACK_X_POS)
            if dist < min_dist and dist < self.DEPOSIT_RANGE:
                min_dist = dist
                target_layer = layer
        
        if target_layer:
            alignment = 1.0 - (min_dist / (self.STACK_WIDTH / 2))
            alignment = max(0, alignment)
            
            color = self.COLOR_GREEN if alignment > 0.5 else self.COLOR_RED
            alpha_color = (*color, int(alignment * 150 + 50))
            
            # Draw a glowing box in the target zone
            indicator_rect = pygame.Rect(0, 0, self.STACK_WIDTH, self.LAYER_HEIGHT + 20)
            indicator_rect.center = (self.STACK_X_POS, self.BELT_Y_POS - self.LAYER_HEIGHT / 2)
            
            s = pygame.Surface(indicator_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, alpha_color, s.get_rect(), border_radius=8)
            self.screen.blit(s, indicator_rect.topleft)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 40.0))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['lifespan'] / 40.0))
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (size, size), size)
                self.screen.blit(s, (int(p['x'] - size), int(p['y'] - size)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Layers Stacked
        self._render_text(f"Layers: {len(self.stacked_layers)} / {self.WIN_CONDITION_LAYERS}", 10, 10, self.font_main, self.COLOR_TEXT)
        
        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.TARGET_FPS
        time_color = self.COLOR_RED if time_left < 10 else self.COLOR_TEXT
        self._render_text(f"Time: {max(0, time_left):.1f}s", self.SCREEN_WIDTH - 150, 10, self.font_main, time_color)

        # Belt Speed
        speed_text = f"Belt Speed: {self.belt_speed:.1f}x"
        self._render_text(speed_text, self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50, self.font_small, self.COLOR_TEXT, center=True)
        
        # Sync Bonus
        is_sync_possible = self.steps - self.last_stack_time < self.SYNC_BONUS_TIME
        if self.sync_bonus_flash_timer > 0:
            color = self.COLOR_GREEN if (self.sync_bonus_flash_timer // 3) % 2 == 0 else self.COLOR_TEXT
            self._render_text("SYNC BONUS!", self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 25, self.font_bonus, color, center=True)
        elif is_sync_possible:
            color = self.COLOR_GREEN
            self._render_text("Sync Active", self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 25, self.font_bonus, color, center=True)

    def close(self):
        pygame.quit()

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


if __name__ == "__main__":
    env = GameEnv(render_mode="human")
    
    # --- Manual Play ---
    # This block allows you to play the game manually.
    # Use Left/Right arrow keys to control speed, Space to drop layers.
    
    obs, info = env.reset()
    done = False
    
    # Override pygame screen for direct display
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Terraform Stacker")
    clock = pygame.time.Clock()
    
    total_reward = 0.0
    
    while not done:
        # Action defaults
        movement = 0 # none
        space = 0 # released
        shift = 0 # released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        if keys[pygame.K_RIGHT]:
            movement = 4
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.metadata["render_fps"])

    print(f"Game Over! Final Score: {total_reward:.2f}")
    print(f"Info: {info}")
    
    env.close()