
# Generated: 2025-08-27T17:51:30.845627
# Source Brief: brief_01662.md
# Brief Index: 1662

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use ← and → to move, and ↑ to jump. Reach the green flag to win!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced procedural platformer. Navigate treacherous terrain, avoid enemies, and reach the flag before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    W, H = 640, 400
    FPS = 30
    
    # Colors
    COLOR_BG_TOP = (100, 149, 237)  # Cornflower Blue
    COLOR_BG_BOTTOM = (0, 0, 128)      # Navy Blue
    COLOR_PLATFORM = (105, 105, 105) # Dim Gray
    COLOR_PLATFORM_OUTLINE = (50, 50, 50)
    COLOR_PLAYER = (65, 105, 225)    # Royal Blue
    COLOR_PLAYER_OUTLINE = (255, 255, 255) # White
    COLOR_ENEMY = (220, 20, 60)      # Crimson
    COLOR_ENEMY_EYE = (255, 255, 255)
    COLOR_FLAG = (0, 200, 0)         # Green
    COLOR_FLAGPOLE = (139, 69, 19)   # Saddle Brown
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0)

    # Physics
    GRAVITY = 0.8
    PLAYER_ACCEL = 1.2
    PLAYER_FRICTION = -0.15
    PLAYER_MAX_SPEED_X = 6
    JUMP_STRENGTH = -15
    MAX_FALL_SPEED = 20

    # Game Rules
    MAX_LIVES = 3
    LEVEL_TIME_SECONDS = 60
    MAX_STEPS = 10000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 36)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 0
        self.level = 0
        self.time_remaining = 0
        self.enemy_base_speed = 1.0

        self.player_pos = None
        self.player_vel = None
        self.on_ground = False
        
        self.platforms = []
        self.enemies = []
        self.flag_rect = None
        self.level_width = 0
        self.camera_x = 0
        
        self.last_dist_to_flag = 0.0

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.MAX_LIVES
        self.level = 1
        self.enemy_base_speed = 1.0
        
        self._generate_level()
        self._reset_player_and_time()

        return self._get_observation(), self._get_info()

    def _reset_player_and_time(self):
        self.player_pos = pygame.math.Vector2(100, self.H - 100)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.on_ground = False
        self.time_remaining = self.LEVEL_TIME_SECONDS * self.FPS
        self.last_dist_to_flag = abs(self.player_pos.x - self.flag_rect.centerx)

    def _generate_level(self):
        self.platforms.clear()
        self.enemies.clear()

        # Start platform
        start_platform = pygame.Rect(0, self.H - 40, 250, 40)
        self.platforms.append(start_platform)
        
        current_x = start_platform.right
        current_y = start_platform.y
        
        # Procedurally generate platforms
        for _ in range(15):
            gap = self.np_random.uniform(80, 160)
            width = self.np_random.uniform(100, 300)
            y_change = self.np_random.uniform(-100, 100)
            
            current_x += gap
            next_y = np.clip(current_y + y_change, 150, self.H - 40)
            
            platform_rect = pygame.Rect(current_x, next_y, width, 200)
            self.platforms.append(platform_rect)

            # Add enemies on some platforms
            if self.np_random.random() < 0.4 and width > 120:
                enemy_y = platform_rect.top - 15
                enemy_x = platform_rect.left + 30
                self.enemies.append({
                    'pos': pygame.math.Vector2(enemy_x, enemy_y),
                    'start_x': platform_rect.left,
                    'end_x': platform_rect.right,
                    'dir': 1
                })

            current_x = platform_rect.right
            current_y = platform_rect.y
        
        # Final platform and flag
        final_platform = self.platforms[-1]
        self.flag_rect = pygame.Rect(final_platform.right - 50, final_platform.top - 50, 10, 50)
        self.level_width = final_platform.right + 100

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Update game clock and state ---
        self.clock.tick(self.FPS)
        self.steps += 1
        self.time_remaining -= 1
        reward = 0

        # --- Handle player actions ---
        movement = action[0]
        self._handle_input(movement)

        # --- Update game logic ---
        self._update_player()
        self._update_enemies()
        
        # --- Handle collisions and events ---
        reward += self._handle_collisions()
        
        # --- Check for terminal conditions ---
        terminated = False
        if self.player_pos.y > self.H + 50: # Fell off screen
            # sound: fall.wav
            reward += self._handle_death()
        elif self.time_remaining <= 0: # Out of time
            # sound: timeout.wav
            reward += self._handle_death()
        
        if self.lives <= 0:
            self.game_over = True
            terminated = True
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            reward -= 10 # Penalty for running out of steps

        # --- Calculate distance-based reward ---
        dist_to_flag = abs(self.player_pos.x - self.flag_rect.centerx)
        if dist_to_flag < self.last_dist_to_flag:
            reward += 0.1
        else:
            reward -= 0.01
        self.last_dist_to_flag = dist_to_flag
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        # movement: 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 3: # Left
            self.player_vel.x -= self.PLAYER_ACCEL
        elif movement == 4: # Right
            self.player_vel.x += self.PLAYER_ACCEL
        
        # Apply friction
        self.player_vel.x += self.player_vel.x * self.PLAYER_FRICTION
        
        # Clamp horizontal speed
        self.player_vel.x = np.clip(self.player_vel.x, -self.PLAYER_MAX_SPEED_X, self.PLAYER_MAX_SPEED_X)
        if abs(self.player_vel.x) < 0.1: self.player_vel.x = 0

        if movement == 1 and self.on_ground: # Jump
            # sound: jump.wav
            self.player_vel.y = self.JUMP_STRENGTH
            self.on_ground = False

    def _update_player(self):
        # Apply gravity
        self.player_vel.y += self.GRAVITY
        self.player_vel.y = min(self.player_vel.y, self.MAX_FALL_SPEED)

        # Move horizontally
        self.player_pos.x += self.player_vel.x
        player_rect = self._get_player_rect()

        for plat in self.platforms:
            if player_rect.colliderect(plat):
                if self.player_vel.x > 0: # Moving right
                    player_rect.right = plat.left
                elif self.player_vel.x < 0: # Moving left
                    player_rect.left = plat.right
                self.player_pos.x = player_rect.x
                self.player_vel.x = 0
        
        # Move vertically
        self.player_pos.y += self.player_vel.y
        player_rect = self._get_player_rect()
        self.on_ground = False

        for plat in self.platforms:
            if player_rect.colliderect(plat):
                if self.player_vel.y > 0: # Moving down
                    player_rect.bottom = plat.top
                    self.on_ground = True
                    self.player_vel.y = 0
                elif self.player_vel.y < 0: # Moving up
                    player_rect.top = plat.bottom
                    self.player_vel.y = 0
                self.player_pos.y = player_rect.y
        
        # Prevent player from going off left edge
        self.player_pos.x = max(self.player_pos.x, 0)

    def _update_enemies(self):
        for enemy in self.enemies:
            speed = self.enemy_base_speed
            enemy['pos'].x += speed * enemy['dir']
            if enemy['pos'].x <= enemy['start_x'] + 15 or enemy['pos'].x >= enemy['end_x'] - 15:
                enemy['dir'] *= -1

    def _handle_collisions(self):
        player_rect = self._get_player_rect()
        reward = 0

        # Enemy collisions
        for enemy in self.enemies:
            enemy_rect = pygame.Rect(enemy['pos'].x - 15, enemy['pos'].y - 15, 30, 30)
            if player_rect.colliderect(enemy_rect):
                # sound: player_hit.wav
                reward += self._handle_death()
                break # only one death per frame

        # Flag collision
        if player_rect.colliderect(self.flag_rect):
            # sound: level_win.wav
            reward += 60 # +10 event, +50 goal
            self.level += 1
            self.enemy_base_speed += 0.05
            self._generate_level()
            self._reset_player_and_time()
        
        return reward
    
    def _handle_death(self):
        self.lives -= 1
        if self.lives > 0:
            self._reset_player_and_time()
        return -5 # Penalty for losing a life

    def _get_player_rect(self):
        return pygame.Rect(self.player_pos.x, self.player_pos.y, 30, 30)

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "level": self.level,
            "time_remaining": int(self.time_remaining / self.FPS),
        }
    
    def _render_all(self):
        # --- Camera ---
        self.camera_x = max(0, min(self.player_pos.x - self.W / 3, self.level_width - self.W))

        # --- Background ---
        for y in range(self.H):
            ratio = y / self.H
            color = (
                self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio,
                self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio,
                self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.W, y))

        # --- Game Elements (with camera offset) ---
        self._render_platforms()
        self._render_enemies()
        self._render_flag()
        self._render_player()
        
        # --- UI (fixed position) ---
        self._render_ui()

    def _render_platforms(self):
        for plat in self.platforms:
            cam_plat = plat.move(-self.camera_x, 0)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, cam_plat, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_OUTLINE, cam_plat, width=2, border_radius=3)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = enemy['pos']
            radius = 15 + 2 * math.sin(self.steps * 0.1)
            cam_pos = (int(pos.x - self.camera_x), int(pos.y))
            pygame.gfxdraw.filled_circle(self.screen, cam_pos[0], cam_pos[1], int(radius), self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, cam_pos[0], cam_pos[1], int(radius), self.COLOR_ENEMY)
            
            # Eyes
            eye_dir = -1 if enemy['dir'] > 0 else 1
            eye_x = cam_pos[0] + 5 * eye_dir
            pygame.draw.circle(self.screen, self.COLOR_ENEMY_EYE, (eye_x, cam_pos[1]-2), 3)

    def _render_flag(self):
        cam_flag = self.flag_rect.move(-self.camera_x, 0)
        pole_rect = pygame.Rect(cam_flag.x, cam_flag.y, 4, cam_flag.height)
        flag_poly = [(cam_flag.x+4, cam_flag.y), (cam_flag.x+4, cam_flag.y+20), (cam_flag.x+24, cam_flag.y+10)]
        
        pygame.draw.rect(self.screen, self.COLOR_FLAGPOLE, pole_rect)
        pygame.draw.polygon(self.screen, self.COLOR_FLAG, flag_poly)

    def _render_player(self):
        player_rect = self._get_player_rect().move(-self.camera_x, 0)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect, width=2, border_radius=3)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
            surface = font.render(text, True, color)
            self.screen.blit(surface, pos)

        # Lives
        heart_char = "♥"
        lives_text = heart_char * self.lives
        draw_text(lives_text, self.font_large, self.COLOR_ENEMY, (20, 10))

        # Time
        time_str = f"Time: {int(self.time_remaining / self.FPS):02d}"
        draw_text(time_str, self.font_small, self.COLOR_TEXT, (self.W - 150, 20))
        
        # Level
        level_str = f"Level: {self.level}"
        draw_text(level_str, self.font_small, self.COLOR_TEXT, (self.W / 2 - 50, 20))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This block allows a human to play the game.
    # It requires pygame to be installed with video support.
    try:
        import getch
        
        # Re-initialize pygame with a display
        pygame.display.init()
        pygame.display.set_caption("Procedural Platformer")
        display_screen = pygame.display.set_mode((GameEnv.W, GameEnv.H))

        obs, info = env.reset()
        terminated = False
        
        # Key mapping
        key_map = {
            'w': 1, 'KEY_UP': 1,
            'a': 3, 'KEY_LEFT': 3,
            'd': 4, 'KEY_RIGHT': 4,
        }
        
        action = np.array([0, 0, 0]) # Default no-op action
        
        print(GameEnv.user_guide)
        
        while not terminated:
            # Non-blocking key check for smooth gameplay
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP: action[0] = 1
                    if event.key == pygame.K_LEFT: action[0] = 3
                    if event.key == pygame.K_RIGHT: action[0] = 4
                if event.type == pygame.KEYUP:
                    # Only reset the movement part of the action
                    if event.key in (pygame.K_UP, pygame.K_LEFT, pygame.K_RIGHT):
                        action[0] = 0

            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render to the display
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated:
                print(f"Game Over! Final Score: {info['score']:.2f}, Level: {info['level']}")
                
    except (ImportError, pygame.error) as e:
        print(f"Could not run interactive demo: {e}")
        print("Running a simple step test instead.")
        obs, info = env.reset()
        terminated = False
        total_reward = 0
        for i in range(2000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                print(f"Episode finished after {i+1} steps. Final info: {info}")
                break
        print(f"Random agent total reward: {total_reward}")

    env.close()