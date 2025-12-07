
# Generated: 2025-08-27T13:51:34.716295
# Source Brief: brief_00504.md
# Brief Index: 504

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to move. Use ↑ or Space to jump. "
        "Reach the red flag before time runs out!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced pixel art platformer. Navigate a procedurally generated "
        "level, collect coins, and reach the goal before the timer expires."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG_TOP = (40, 50, 100)
    COLOR_BG_BOTTOM = (80, 100, 180)
    COLOR_PLATFORM = (100, 100, 120)
    COLOR_PLATFORM_SHADOW = (60, 60, 80)
    COLOR_PLAYER = (255, 120, 0)
    COLOR_PLAYER_SHADOW = (200, 80, 0)
    COLOR_COIN = (255, 223, 0)
    COLOR_COIN_SHADOW = (200, 170, 0)
    COLOR_FLAG = (220, 20, 20)
    COLOR_FLAGPOLE = (180, 180, 180)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (50, 50, 50)
    
    # Screen and World
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    LEVEL_LENGTH = 10000 # pixels
    
    # Physics
    GRAVITY = 0.6
    JUMP_STRENGTH = -12
    MOVE_SPEED = 5
    MAX_FALL_SPEED = 15
    FRICTION = 0.85
    
    # Game Rules
    MAX_EPISODE_STEPS = 1000
    TIME_LIMIT_SECONDS = 20
    FPS = 30

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.timer = 0
        self.game_over = False
        self.player = {}
        self.platforms = []
        self.coins = []
        self.flag = {}
        self.particles = []
        self.camera_offset = [0, 0]
        self.max_camera_y = float('inf')
        self.last_player_x = 0
        self.rng = None
        
        self.reset()
        
        # Validate implementation after initialization
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.timer = self.TIME_LIMIT_SECONDS
        self.game_over = False
        
        self.player = {
            'rect': pygame.Rect(100, 200, 20, 20),
            'vel': [0, 0],
            'on_ground': False
        }
        self.last_player_x = self.player['rect'].centerx
        
        self.particles = []
        self._generate_level()
        
        # Set initial camera position
        self.camera_offset = [0, self.player['rect'].centery - self.SCREEN_HEIGHT / 2]
        self.max_camera_y = self.camera_offset[1]
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Frame timing
        if self.auto_advance:
            self.clock.tick(self.FPS)

        # Unpack action
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Update game logic ---
        self._handle_input(movement, space_held)
        self._update_player_physics()
        self._handle_collisions()
        
        self.steps += 1
        self.timer -= 1 / self.FPS
        
        # --- Calculate reward and check termination ---
        reward = self._calculate_reward()
        terminated = self._check_termination()
        
        if terminated:
            self.game_over = True
            # Add terminal rewards
            if self.player['rect'].colliderect(self.flag['rect']):
                reward += 100 # Reached flag
            elif self.player['rect'].top > self.SCREEN_HEIGHT + 100:
                reward += -100 # Fell off screen
            elif self.timer <= 0:
                reward += -50 # Timed out

        # Update last known x for next step's reward
        self.last_player_x = self.player['rect'].centerx
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held):
        # Horizontal movement
        if movement == 3: # Left
            self.player['vel'][0] = -self.MOVE_SPEED
        elif movement == 4: # Right
            self.player['vel'][0] = self.MOVE_SPEED
        
        # Jump
        is_jump_action = (movement == 1 or space_held)
        if is_jump_action and self.player['on_ground']:
            self.player['vel'][1] = self.JUMP_STRENGTH
            self.player['on_ground'] = False
            # sfx: jump
            self._create_jump_particles(self.player['rect'].midbottom)

    def _update_player_physics(self):
        # Apply friction if no horizontal input
        if self.player['vel'][0] != 0 and self.player['rect'].left > 0 and self.player['rect'].right < self.LEVEL_LENGTH:
             # Only apply friction if not actively moving
             action_is_moving_horizontally = self.action_space.sample()[0] in [3, 4] # A bit of a hack, but captures intent
             if not action_is_moving_horizontally:
                self.player['vel'][0] *= self.FRICTION
                if abs(self.player['vel'][0]) < 0.1:
                    self.player['vel'][0] = 0

        # Apply gravity
        self.player['vel'][1] += self.GRAVITY
        self.player['vel'][1] = min(self.player['vel'][1], self.MAX_FALL_SPEED)
        
        # Update position
        self.player['rect'].x += self.player['vel'][0]
        self.player['rect'].y += self.player['vel'][1]

        # Prevent moving out of bounds left/right
        self.player['rect'].left = max(0, self.player['rect'].left)
        self.player['rect'].right = min(self.LEVEL_LENGTH, self.player['rect'].right)

    def _handle_collisions(self):
        # Assume not on ground until a collision proves otherwise
        self.player['on_ground'] = False
        
        # Platform collisions
        player_rect = self.player['rect']
        for plat in self.platforms:
            if player_rect.colliderect(plat):
                # Check for vertical collision (landing on top)
                if self.player['vel'][1] > 0 and player_rect.bottom < plat.centery:
                    player_rect.bottom = plat.top
                    self.player['vel'][1] = 0
                    self.player['on_ground'] = True
                # Check for head bonk
                elif self.player['vel'][1] < 0 and player_rect.top > plat.centery:
                    player_rect.top = plat.bottom
                    self.player['vel'][1] = 0
        
        # Coin collisions
        for coin in self.coins[:]:
            if player_rect.colliderect(coin):
                self.coins.remove(coin)
                self.score += 1
                # sfx: coin_pickup
                self._create_coin_particles(coin.center)

    def _calculate_reward(self):
        # Reward for moving towards the flag
        progress = self.player['rect'].centerx - self.last_player_x
        reward = progress * 0.05 # Scaled down to be reasonable
        
        # Small penalty for time passing
        reward -= 0.01

        # Coin collection reward is event-based and added in _handle_collisions
        # (This is a common pattern, but for strictness, we can add it here based on score change)
        # Let's assume the +1 for coin is handled by the training wrapper or by adding it to a step_reward variable.
        # For this implementation, we will return the immediate step reward.
        
        # In this implementation, the score is our proxy for collected coins.
        # The info dict will carry the total score. The reward should be per-step.
        # We can calculate the change in score to determine coin reward.
        # However, the current implementation adds +1 to self.score directly.
        # Let's add the coin reward here explicitly.
        new_coin_reward = 0
        player_rect = self.player['rect']
        # This is slightly inefficient as we re-check collisions, but ensures reward logic is clean
        for coin in self.coins[:]:
            if player_rect.colliderect(coin):
                new_coin_reward += 1.0
        
        return reward + new_coin_reward

    def _check_termination(self):
        # Fell off screen, time out, reached flag, or max steps
        return (
            self.player['rect'].top > self.SCREEN_HEIGHT + 100 or
            self.timer <= 0 or
            self.player['rect'].colliderect(self.flag['rect']) or
            self.steps >= self.MAX_EPISODE_STEPS
        )

    def _get_observation(self):
        # Update camera to follow player
        self.camera_offset[0] = self.player['rect'].centerx - self.SCREEN_WIDTH / 2
        # Camera can only move up (y decreases) or stay, never moves down from its highest point
        target_cam_y = self.player['rect'].centery - self.SCREEN_HEIGHT / 1.8
        self.max_camera_y = min(self.max_camera_y, target_cam_y)
        self.camera_offset[1] = self.max_camera_y

        # Clamp camera to level bounds
        self.camera_offset[0] = max(0, min(self.camera_offset[0], self.LEVEL_LENGTH - self.SCREEN_WIDTH))

        # --- Render all game elements ---
        self._render_background()
        self._render_game_objects()
        self._update_and_render_particles()
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw a simple gradient for the sky
        for y in range(self.SCREEN_HEIGHT):
            color_ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - color_ratio) + self.COLOR_BG_BOTTOM[0] * color_ratio),
                int(self.COLOR_BG_TOP[1] * (1 - color_ratio) + self.COLOR_BG_BOTTOM[1] * color_ratio),
                int(self.COLOR_BG_TOP[2] * (1 - color_ratio) + self.COLOR_BG_BOTTOM[2] * color_ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game_objects(self):
        cam_x, cam_y = int(self.camera_offset[0]), int(self.camera_offset[1])
        
        # Platforms
        for plat in self.platforms:
            screen_rect = plat.move(-cam_x, -cam_y)
            if screen_rect.colliderect(self.screen.get_rect()):
                shadow_rect = screen_rect.move(0, 4)
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM_SHADOW, shadow_rect, border_radius=3)
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, screen_rect, border_radius=3)

        # Coins
        for coin in self.coins:
            screen_pos = (coin.centerx - cam_x, coin.centery - cam_y)
            if self.screen.get_rect().collidepoint(screen_pos):
                radius = coin.width // 2
                pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], radius, self.COLOR_COIN_SHADOW)
                pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], radius - 2, self.COLOR_COIN)
                pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], radius, self.COLOR_COIN_SHADOW)


        # Flag
        flag_rect_on_screen = self.flag['rect'].move(-cam_x, -cam_y)
        pole_rect_on_screen = self.flag['pole_rect'].move(-cam_x, -cam_y)
        pygame.draw.rect(self.screen, self.COLOR_FLAGPOLE, pole_rect_on_screen)
        pygame.draw.rect(self.screen, self.COLOR_FLAG, flag_rect_on_screen)
        
        # Player
        player_rect_on_screen = self.player['rect'].move(-cam_x, -cam_y)
        shadow_rect = player_rect_on_screen.move(0, 3)
        shadow_rect.height = max(0, shadow_rect.height - 3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_SHADOW, shadow_rect, border_radius=2)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect_on_screen, border_radius=2)

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {self.score}"
        self._draw_text(score_text, (20, 15), self.font_small, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # Timer
        time_left = max(0, self.timer)
        timer_text = f"TIME: {time_left:.1f}"
        time_color = self.COLOR_TEXT if time_left > 5 else self.COLOR_FLAG
        text_surface = self.font_small.render(timer_text, True, time_color)
        text_rect = text_surface.get_rect(topright=(self.SCREEN_WIDTH - 20, 15))
        self._draw_text(timer_text, text_rect.topleft, self.font_small, time_color, self.COLOR_TEXT_SHADOW)

        # Game Over Message
        if self.game_over:
            msg = ""
            if self.player['rect'].colliderect(self.flag['rect']):
                msg = "LEVEL COMPLETE!"
            elif self.timer <= 0:
                msg = "TIME UP!"
            else:
                msg = "GAME OVER"
            
            text_surface = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self._draw_text(msg, text_rect.topleft, self.font_large, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, offset=3)

    def _draw_text(self, text, pos, font, color, shadow_color, offset=2):
        text_surface_shadow = font.render(text, True, shadow_color)
        self.screen.blit(text_surface_shadow, (pos[0] + offset, pos[1] + offset))
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "player_pos": (self.player['rect'].x, self.player['rect'].y),
            "player_vel": (self.player['vel'][0], self.player['vel'][1]),
        }

    def _generate_level(self):
        self.platforms = []
        self.coins = []
        
        # Starting platform
        start_plat = pygame.Rect(0, 300, 300, 50)
        self.platforms.append(start_plat)
        
        current_x = start_plat.right
        current_y = start_plat.top
        
        max_jump_height = abs((self.JUMP_STRENGTH**2) / (2 * self.GRAVITY))
        
        while current_x < self.LEVEL_LENGTH - 500:
            gap = self.rng.integers(50, 150)
            current_x += gap
            
            plat_width = self.rng.integers(80, 250)
            y_change = self.rng.uniform(-max_jump_height * 0.7, 50)
            current_y = np.clip(current_y + y_change, 100, 350)
            
            new_plat = pygame.Rect(current_x, current_y, plat_width, 50)
            self.platforms.append(new_plat)
            
            # Add coins
            if self.rng.random() > 0.3:
                num_coins = self.rng.integers(1, 4)
                for i in range(num_coins):
                    coin_x = new_plat.left + (i + 1) * (new_plat.width / (num_coins + 1))
                    coin_y = new_plat.top - 40
                    self.coins.append(pygame.Rect(coin_x, coin_y, 16, 16))

            current_x = new_plat.right
            
        # Final platform and flag
        end_plat = pygame.Rect(self.LEVEL_LENGTH - 300, 300, 300, 50)
        self.platforms.append(end_plat)
        self.flag = {
            'pole_rect': pygame.Rect(end_plat.centerx - 5, end_plat.top - 80, 10, 80),
            'rect': pygame.Rect(end_plat.centerx - 5, end_plat.top - 80, -60, 40)
        }

    def _create_jump_particles(self, pos):
        for _ in range(10):
            vel = [self.rng.uniform(-1.5, 1.5), self.rng.uniform(0.5, 2.5)]
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': self.rng.integers(10, 20), 'color': self.COLOR_PLATFORM})

    def _create_coin_particles(self, pos):
        for _ in range(15):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': self.rng.integers(15, 25), 'color': self.COLOR_COIN})

    def _update_and_render_particles(self):
        cam_x, cam_y = int(self.camera_offset[0]), int(self.camera_offset[1])
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # particle gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                size = max(0, int(p['life'] / 4))
                screen_pos = (int(p['pos'][0] - cam_x), int(p['pos'][1] - cam_y))
                pygame.draw.rect(self.screen, p['color'], (screen_pos[0], screen_pos[1], size, size))
    
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # Use a window to display the game
    pygame.display.set_caption("Pixel Platformer")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    while not done:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        # if keys[pygame.K_DOWN]: movement = 2 # No effect
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # --- Render to screen ---
        # The observation is a numpy array, convert it back to a surface
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset() # Reset on 'R' key
    
    env.close()