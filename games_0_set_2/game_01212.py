import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ←→ to run and ↑ or Space to jump. Collect all the coins and reach the flag!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced pixel art platformer. Race against the clock to collect coins and reach the flag across three challenging levels."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Game
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30

    # Colors
    COLOR_BG = (135, 206, 235)  # Sky Blue
    COLOR_PLAYER = (255, 50, 50)  # Bright Red
    COLOR_PLATFORM = (139, 69, 19)  # Brown
    COLOR_PLATFORM_TOP = (160, 82, 45)  # Lighter Brown
    COLOR_COIN = (255, 215, 0)  # Gold
    COLOR_FLAG_POLE = (192, 192, 192)  # Silver
    COLOR_FLAG = (0, 200, 0)  # Green
    COLOR_UI_BG = (0, 0, 0, 128)
    COLOR_UI_TEXT = (255, 255, 255)

    # Physics
    GRAVITY = 0.8
    JUMP_STRENGTH = -14
    PLAYER_SPEED = 6
    PLAYER_FRICTION = 0.8
    TERMINAL_VELOCITY = 15

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)

        # Game state variables
        self.player_pos = None
        self.player_vel = None
        self.player_rect = None
        self.player_size = [24, 32]
        self.on_ground = False
        self.can_jump = True

        self.platforms = []
        self.coins = []
        self.flag_pos = None
        self.level_bounds = None

        self.camera_x = 0
        self.particles = []

        self.level = 1
        self.score = 0
        self.time_remaining = 0
        self.steps = 0
        self.game_over = False
        self.victory = False

        self._level_data = self._get_level_definitions()

        # Initialize state variables
        self.reset()

        # Run validation check
        # self.validate_implementation() # Optional validation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        self.level = 1
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.victory = False

        self._load_level(self.level)

        self.player_pos = [100.0, 200.0]
        self.player_vel = [0.0, 0.0]
        # FIX: Initialize player_rect in reset to avoid NoneType error on first render.
        self.player_rect = pygame.Rect(int(self.player_pos[0]), int(self.player_pos[1]), self.player_size[0], self.player_size[1])
        self.on_ground = False
        self.can_jump = True
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- 1. Handle Input ---
        target_vx = 0
        if movement == 3:  # Left
            target_vx = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            target_vx = self.PLAYER_SPEED

        self.player_vel[0] = target_vx if target_vx != 0 else self.player_vel[0] * self.PLAYER_FRICTION

        jump_pressed = (movement == 1) or space_held
        if jump_pressed and self.on_ground and self.can_jump:
            self.player_vel[1] = self.JUMP_STRENGTH
            self.on_ground = False
            self.can_jump = False
            # Sound: Player Jump
            self._spawn_particles(self.player_rect.midbottom, 5, (200, 200, 200))  # Dust kick-up

        if not jump_pressed:
            self.can_jump = True

        # --- 2. Update Physics & Position ---
        self.player_vel[1] += self.GRAVITY
        self.player_vel[1] = min(self.player_vel[1], self.TERMINAL_VELOCITY)

        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]

        self.player_rect = pygame.Rect(int(self.player_pos[0]), int(self.player_pos[1]), self.player_size[0], self.player_size[1])

        # --- 3. Collision Detection ---
        self.on_ground = False

        # Level bounds collision
        if self.player_rect.left < self.level_bounds.left:
            self.player_rect.left = self.level_bounds.left
            self.player_pos[0] = self.player_rect.x
            self.player_vel[0] = 0
        if self.player_rect.right > self.level_bounds.right:
            self.player_rect.right = self.level_bounds.right
            self.player_pos[0] = self.player_rect.x
            self.player_vel[0] = 0

        for plat in self.platforms:
            if self.player_rect.colliderect(plat):
                # Vertical collision
                if self.player_vel[1] > 0 and self.player_rect.bottom < plat.centery:
                    self.player_rect.bottom = plat.top
                    self.player_pos[1] = self.player_rect.y
                    if not self.on_ground:  # First frame of landing
                        # Sound: Player Land
                        self._spawn_particles(self.player_rect.midbottom, 3, (160, 82, 45))
                    self.on_ground = True
                    self.player_vel[1] = 0
                # Horizontal collision
                elif self.player_vel[0] > 0 and self.player_rect.right < plat.centerx:
                    self.player_rect.right = plat.left
                    self.player_pos[0] = self.player_rect.x
                    self.player_vel[0] = 0
                elif self.player_vel[0] < 0 and self.player_rect.left > plat.centerx:
                    self.player_rect.left = plat.right
                    self.player_pos[0] = self.player_rect.x
                    self.player_vel[0] = 0
                # Bonking head
                elif self.player_vel[1] < 0:
                    self.player_rect.top = plat.bottom
                    self.player_pos[1] = self.player_rect.y
                    self.player_vel[1] = 0

        # --- 4. Game Logic & Rewards ---
        reward = -0.01  # Time penalty

        # Coin collection
        for coin in self.coins:
            if not coin['collected'] and self.player_rect.colliderect(coin['rect']):
                coin['collected'] = True
                self.score += 1
                reward += 1
                # Sound: Coin Collect
                self._spawn_particles(coin['rect'].center, 10, self.COLOR_COIN)

        # Flag collision (level complete)
        if self.player_rect.colliderect(pygame.Rect(self.flag_pos[0], self.flag_pos[1], 10, 100)):
            if self.level < 3:
                self.level += 1
                reward += 10
                self._load_level(self.level)
                self.player_pos = [100.0, 200.0]  # Reset position for new level
                self.player_vel = [0.0, 0.0]
                self.player_rect.topleft = (int(self.player_pos[0]), int(self.player_pos[1]))
            else:
                self.victory = True
                self.game_over = True
                reward += 100
                # Sound: Victory

        # --- 5. Update Timers & Particles ---
        self.time_remaining -= 1
        self.steps += 1

        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # --- 6. Check Termination ---
        terminated = False
        if self.player_pos[1] > self.SCREEN_HEIGHT + 100:  # Fell off
            self.game_over = True
            # Sound: Player Fall/Fail
        if self.time_remaining <= 0:
            self.game_over = True
            # Sound: Time Up

        if self.game_over or self.victory:
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        # Update camera to follow player
        self.camera_x += (self.player_pos[0] - self.camera_x - self.SCREEN_WIDTH / 2) * 0.1
        self.camera_x = max(self.level_bounds.left, min(self.camera_x, self.level_bounds.right - self.SCREEN_WIDTH))

        # Render background
        self._render_background()

        # Render game elements
        self._render_game_objects()

        # Render UI overlay
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        # Parallax hills
        for i in range(3):
            speed = 0.2 + i * 0.2
            offset = -self.camera_x * speed
            for j in range(math.ceil(self.level_bounds.width / 400) + 2):
                hill_x = j * 400 + (offset % 400) - 400
                hill_y = 250 + i * 40
                color = (
                    max(0, self.COLOR_BG[0] - 40 - i * 15),
                    max(0, self.COLOR_BG[1] - 40 - i * 15),
                    max(0, self.COLOR_BG[2] - 30 - i * 15)
                )
                pygame.gfxdraw.filled_ellipse(self.screen, int(hill_x), int(hill_y), 200, 80 - i * 10, color)

    def _render_game_objects(self):
        cam_x = int(self.camera_x)

        # Platforms
        for plat in self.platforms:
            p_rect = plat.move(-cam_x, 0)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, p_rect)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_TOP, (p_rect.x, p_rect.y, p_rect.width, 3))

        # Coins
        for coin in self.coins:
            if not coin['collected']:
                c_rect = coin['rect'].move(-cam_x, 0)
                # Spinning animation
                width = int(abs(math.sin((self.steps + coin['offset']) * 0.1) * coin['rect'].width))
                anim_rect = pygame.Rect(0, 0, width, coin['rect'].height)
                anim_rect.center = c_rect.center
                pygame.draw.ellipse(self.screen, self.COLOR_COIN, anim_rect)

        # Flag
        flag_pole_rect = pygame.Rect(self.flag_pos[0] - cam_x, self.flag_pos[1], 5, 100)
        pygame.draw.rect(self.screen, self.COLOR_FLAG_POLE, flag_pole_rect)
        flag_points = [
            (flag_pole_rect.right, flag_pole_rect.top),
            (flag_pole_rect.right + 40, flag_pole_rect.top + 20),
            (flag_pole_rect.right, flag_pole_rect.top + 40)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_FLAG, flag_points)

        # Particles
        for p in self.particles:
            pos = (int(p['pos'][0] - cam_x), int(p['pos'][1]))
            size = max(1, int(p['life'] / p['max_life'] * 4))
            pygame.draw.rect(self.screen, p['color'], (*pos, size, size))

        # Player
        if self.player_rect:
            p_rect = self.player_rect.move(-cam_x, 0)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, p_rect)
            # Eye for direction
            eye_x = p_rect.centerx + (5 if self.player_vel[0] >= 0 else -5)
            eye_y = p_rect.centery - 5
            pygame.draw.rect(self.screen, (255, 255, 255), (eye_x - 1, eye_y - 1, 3, 3))

    def _render_ui(self):
        ui_surface = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_surface.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surface, (0, 0))

        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 5))

        time_sec = max(0, self.time_remaining // self.FPS)
        time_text = self.font_ui.render(f"TIME: {time_sec}", True, self.COLOR_UI_TEXT)
        time_rect = time_text.get_rect(centerx=self.SCREEN_WIDTH / 2)
        time_rect.y = 5
        self.screen.blit(time_text, time_rect)

        level_text = self.font_ui.render(f"LEVEL: {self.level}", True, self.COLOR_UI_TEXT)
        level_rect = level_text.get_rect(right=self.SCREEN_WIDTH - 10)
        level_rect.y = 5
        self.screen.blit(level_text, level_rect)

        if self.game_over:
            msg = "VICTORY!" if self.victory else "GAME OVER"
            color = self.COLOR_FLAG if self.victory else self.COLOR_PLAYER
            end_font = pygame.font.SysFont("Consolas", 72, bold=True)
            end_text = end_font.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            pygame.draw.rect(self.screen, (0, 0, 0, 180), end_rect.inflate(20, 20))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "time_remaining": self.time_remaining // self.FPS,
        }

    def _load_level(self, level_num):
        self.platforms.clear()
        self.coins.clear()

        level_data = self._level_data[level_num - 1]

        for p in level_data['platforms']:
            self.platforms.append(pygame.Rect(p))

        for i, c in enumerate(level_data['coins']):
            self.coins.append({
                'rect': pygame.Rect(c[0], c[1], 16, 16),
                'collected': False,
                'offset': i * 5  # for animation
            })

        self.flag_pos = level_data['flag']

        min_x = min(p.left for p in self.platforms)
        max_x = max(p.right for p in self.platforms)
        self.level_bounds = pygame.Rect(min_x, 0, max_x - min_x, self.SCREEN_HEIGHT)

        self.time_remaining = level_data['time'] * self.FPS

    def _spawn_particles(self, pos, count, color):
        for _ in range(count):
            self.particles.append({
                'pos': list(pos),
                'vel': [random.uniform(-2, 2), random.uniform(-3, -1)],
                'life': random.randint(10, 20),
                'max_life': 20,
                'color': color
            })

    def _get_level_definitions(self):
        return [
            # Level 1
            {
                'time': 60,
                'platforms': [
                    (0, 350, 800, 50), (900, 300, 200, 50), (1200, 250, 300, 50),
                    (1600, 350, 500, 50), (1800, 250, 100, 20), (2200, 350, 400, 50)
                ],
                'coins': [
                    (300, 320), (600, 320), (950, 270), (1250, 220), (1350, 220),
                    (1700, 320), (1950, 320), (2300, 320)
                ],
                'flag': (2500, 250)
            },
            # Level 2
            {
                'time': 50,
                'platforms': [
                    (0, 350, 500, 50), (650, 300, 150, 20), (900, 250, 150, 20),
                    (1150, 300, 150, 20), (1400, 350, 400, 50), (1600, 250, 50, 20),
                    (1750, 200, 50, 20), (1900, 250, 50, 20), (2100, 350, 500, 50)
                ],
                'coins': [
                    (250, 320), (450, 320), (700, 270), (950, 220), (1200, 270),
                    (1500, 320), (1615, 220), (1765, 170), (1915, 220), (2300, 320)
                ],
                'flag': (2500, 250)
            },
            # Level 3
            {
                'time': 40,
                'platforms': [
                    (0, 350, 300, 50), (450, 320, 50, 20), (600, 280, 50, 20),
                    (750, 240, 50, 20), (900, 280, 50, 20), (1050, 320, 50, 20),
                    (1200, 350, 300, 50), (1300, 250, 100, 20), (1600, 350, 200, 50),
                    (1900, 300, 50, 20), (1900, 200, 50, 20), (2100, 250, 50, 20),
                    (2300, 350, 300, 50)
                ],
                'coins': [
                    (150, 320), (465, 290), (615, 250), (765, 210), (915, 250),
                    (1065, 290), (1350, 220), (1700, 320), (1915, 270), (1915, 170),
                    (2115, 220), (2400, 320)
                ],
                'flag': (2500, 250)
            }
        ]

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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
        assert trunc is False
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # To play the game manually, run this file.
    # Make sure to remove the headless environment variable for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    # pip install gymnasium[classic-control]
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Pixel Platformer")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    while running:
        movement = 0  # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        if keys[pygame.K_RIGHT]:
            movement = 4
        if keys[pygame.K_UP]:
            movement = 1
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(GameEnv.FPS)

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

    pygame.quit()