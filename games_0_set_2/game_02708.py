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
        "Controls: Use ← and → to move, and ↑ to jump. Collect all the coins and reach the green flag!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced retro platformer. Navigate treacherous terrain, collect coins, and reach the end flag on each level before time runs out. Difficulty increases with moving and falling platforms."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400

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

        # Fonts
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
            self.font_level = pygame.font.SysFont("Consolas", 24, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 24)
            self.font_level = pygame.font.SysFont(None, 28)

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_PLATFORM = (100, 110, 130)
        self.COLOR_PLATFORM_MOVING = (130, 110, 150)
        self.COLOR_PLATFORM_FALLING = (130, 130, 100)
        self.COLOR_COIN = (255, 220, 0)
        self.COLOR_FLAG = (50, 200, 50)
        self.COLOR_TEXT = (255, 255, 255)

        # Game physics and constants
        self.GRAVITY = 0.8
        self.PLAYER_JUMP_STRENGTH = -14
        self.PLAYER_SPEED = 6
        self.PLAYER_FRICTION = -0.2
        self.MAX_LEVELS = 3
        self.TIME_PER_LEVEL = 60 * 30  # 60 seconds at 30 fps

        # Initialize state variables
        self.player_rect = None
        self.player_vel = None
        self.on_ground = None
        self.jump_squash = None
        self.camera_x = None
        self.level_width = None
        self.platforms = None
        self.moving_platforms = None
        self.falling_platforms = None
        self.coins = None
        self.end_flag = None
        self.particles = None
        self.current_level = None
        self.time_remaining = None
        self.steps = None
        self.score = None
        self.game_over = None

        # Initialize state
        self.reset()

        # Run validation if needed (optional)
        # self.validate_implementation()

    def _load_level(self, level_num):
        """Sets up the game state for a specific level."""
        self.player_rect = pygame.Rect(100, 200, 24, 24)
        # FIX: Initialize camera_x here, after player_rect is set
        self.camera_x = self.player_rect.centerx - self.SCREEN_WIDTH / 2.0
        self.player_vel = pygame.math.Vector2(0, 0)
        self.on_ground = False
        self.jump_squash = 0
        self.particles = []
        self.time_remaining = self.TIME_PER_LEVEL

        self.platforms = []
        self.moving_platforms = []
        self.falling_platforms = []
        self.coins = []

        # Common starting platform
        self.platforms.append(pygame.Rect(50, 300, 200, 20))

        plat_data = []
        coin_data = []

        if level_num == 1:
            self.level_width = 2000
            plat_data = [(400, 280, 150, 20), (650, 250, 150, 20), (900, 220, 150, 20), (1150, 250, 100, 20), (1350, 280, 150, 20), (1600, 250, 200, 20)]
            coin_data = [(450, 250), (700, 220), (950, 190), (1200, 220), (1400, 250), (1650, 220)]
            self.end_flag = pygame.Rect(1750, 200, 30, 50)

        elif level_num == 2:
            self.level_width = 2500
            plat_data = [(400, 320, 100, 20), (800, 320, 100, 20), (1200, 320, 100, 20), (1600, 250, 150, 20), (2000, 250, 200, 20)]
            coin_data = [(550, 200), (950, 200), (1350, 200), (1675, 220), (2100, 220)]
            # Add moving platforms
            self.moving_platforms.append({'rect': pygame.Rect(500, 250, 100, 20), 'start_x': 500, 'end_x': 750, 'vx': 2})
            self.moving_platforms.append({'rect': pygame.Rect(900, 250, 100, 20), 'start_x': 900, 'end_x': 1150, 'vx': -2})
            self.moving_platforms.append({'rect': pygame.Rect(1300, 250, 100, 20), 'start_x': 1300, 'end_x': 1550, 'vx': 2.5})
            self.end_flag = pygame.Rect(2150, 200, 30, 50)

        elif level_num == 3:
            self.level_width = 3000
            plat_data = [(450, 320, 100, 20), (1300, 280, 150, 20), (2200, 280, 200, 20)]
            coin_data = [(650, 200), (950, 150), (1500, 150), (1800, 180), (2300, 250)]
            # Add falling platforms
            fall_plat_data = [(600, 250, 120, 20), (850, 200, 120, 20), (1100, 180, 120, 20), (1500, 200, 100, 20), (1700, 250, 100, 20), (1900, 220, 100, 20)]
            for x, y, w, h in fall_plat_data:
                self.falling_platforms.append({'rect': pygame.Rect(x, y, w, h), 'state': 'stable', 'timer': 0, 'vy': 0})
            self.end_flag = pygame.Rect(2350, 230, 30, 50)

        for p in plat_data:
            self.platforms.append(pygame.Rect(*p))
        for c in coin_data:
            self.coins.append(pygame.math.Vector2(*c))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_level = 1

        self._load_level(self.current_level)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right

        reward = 0
        self.steps += 1
        self.time_remaining -= 1

        # -- Player Logic --
        # Horizontal movement
        target_vx = 0
        if movement == 3:  # Left
            target_vx = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            target_vx = self.PLAYER_SPEED

        # Smooth acceleration/deceleration
        self.player_vel.x += (target_vx - self.player_vel.x) * 0.5

        # Reward for moving towards the flag
        if self.player_vel.x > 0 and self.player_rect.centerx < self.end_flag.centerx:
            reward += 0.01 * self.player_vel.x

        # Penalty for standing still on ground
        if abs(self.player_vel.x) < 0.5 and self.on_ground:
            reward -= 0.02

        # Vertical movement (gravity)
        self.player_vel.y += self.GRAVITY
        if self.player_vel.y > 15:  # Terminal velocity
            self.player_vel.y = 15

        # Jumping
        if movement == 1 and self.on_ground:
            self.player_vel.y = self.PLAYER_JUMP_STRENGTH
            self.on_ground = False
            self.jump_squash = 10  # For animation

        # -- Update Platforms --
        for plat in self.moving_platforms:
            plat['rect'].x += plat['vx']
            if plat['rect'].left < plat['start_x'] or plat['rect'].right > plat['end_x']:
                plat['vx'] *= -1

        for plat in self.falling_platforms:
            if plat['state'] == 'shaking':
                plat['timer'] -= 1
                if plat['timer'] <= 0:
                    plat['state'] = 'falling'
            elif plat['state'] == 'falling':
                plat['vy'] += self.GRAVITY * 0.5
                plat['rect'].y += plat['vy']

        # -- Collisions and Position Update --
        # Move horizontal
        self.player_rect.x += self.player_vel.x

        all_platforms = self.platforms + [p['rect'] for p in self.moving_platforms] + [p['rect'] for p in self.falling_platforms if p['state'] != 'falling']

        # Horizontal collision
        for plat_rect in all_platforms:
            if self.player_rect.colliderect(plat_rect):
                if self.player_vel.x > 0:  # Moving right
                    self.player_rect.right = plat_rect.left
                elif self.player_vel.x < 0:  # Moving left
                    self.player_rect.left = plat_rect.right
                self.player_vel.x = 0

        # Move vertical
        self.player_rect.y += self.player_vel.y
        self.on_ground = False

        # Vertical collision
        all_platforms = self.platforms + [p['rect'] for p in self.moving_platforms] + [p['rect'] for p in self.falling_platforms if p['state'] != 'falling']
        for plat_rect in all_platforms:
            if self.player_rect.colliderect(plat_rect):
                if self.player_vel.y > 0:  # Moving down
                    self.player_rect.bottom = plat_rect.top
                    self.player_vel.y = 0
                    self.on_ground = True
                    # Check if it's a falling platform
                    for fp in self.falling_platforms:
                        if fp['rect'] == plat_rect and fp['state'] == 'stable':
                            fp['state'] = 'shaking'
                            fp['timer'] = 30  # 1 second
                elif self.player_vel.y < 0:  # Moving up
                    self.player_rect.top = plat_rect.bottom
                    self.player_vel.y = 0

        # -- Game Element Interactions --
        # Coin collection
        collected_coins = []
        for coin_pos in self.coins:
            if self.player_rect.collidepoint(coin_pos.x, coin_pos.y):
                collected_coins.append(coin_pos)
                self.score += 1
                reward += 10
                self._spawn_particles(coin_pos, self.COLOR_COIN, 10)
        self.coins = [c for c in self.coins if c not in collected_coins]

        # -- Update Particles --
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['radius'] -= 0.1

        # -- Level Completion --
        terminated = False
        if self.player_rect.colliderect(self.end_flag):
            if self.current_level < self.MAX_LEVELS:
                self.current_level += 1
                self.score += 10
                reward += 50
                self._load_level(self.current_level)
            else:  # Game won
                self.score += 100
                reward += 100
                self.game_over = True
                terminated = True

        # -- Termination Check --
        if self.player_rect.top > self.SCREEN_HEIGHT:  # Fell off screen
            reward -= 100
            self.game_over = True
            terminated = True

        if self.time_remaining <= 0:
            reward -= 100
            self.game_over = True
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append({
                'pos': list(pos),
                'vel': [random.uniform(-2, 2), random.uniform(-2, 2)],
                'radius': random.uniform(2, 5),
                'lifespan': random.randint(15, 30),
                'color': color
            })

    def _get_observation(self):
        # Update camera
        target_camera_x = self.player_rect.centerx - self.SCREEN_WIDTH / 2
        # Smooth camera movement
        self.camera_x += (target_camera_x - self.camera_x) * 0.1
        self.camera_x = max(0, min(self.camera_x, self.level_width - self.SCREEN_WIDTH))

        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        offset_x = -int(self.camera_x)

        # Draw platforms
        all_platform_rects = [p.copy() for p in self.platforms]
        moving_platform_rects = [p['rect'].copy() for p in self.moving_platforms]
        falling_platform_data = [{'rect': p['rect'].copy(), 'state': p['state']} for p in self.falling_platforms]

        for plat_rect in all_platform_rects:
            draw_rect = plat_rect.move(offset_x, 0)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, draw_rect, border_radius=3)

        for plat_rect in moving_platform_rects:
            draw_rect = plat_rect.move(offset_x, 0)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_MOVING, draw_rect, border_radius=3)

        for fp_data in falling_platform_data:
            plat_rect = fp_data['rect']
            if fp_data['state'] == 'shaking':
                # Vibrate effect for rendering only
                draw_rect = plat_rect.move(offset_x + random.randint(-1, 1), random.randint(-1, 1))
            else:
                draw_rect = plat_rect.move(offset_x, 0)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_FALLING, draw_rect, border_radius=3)

        # Draw coins
        for i, coin_pos in enumerate(self.coins):
            y_bob = math.sin(self.steps / 10 + i) * 3
            pygame.gfxdraw.filled_circle(self.screen, int(coin_pos.x + offset_x), int(coin_pos.y + y_bob), 8, self.COLOR_COIN)
            pygame.gfxdraw.aacircle(self.screen, int(coin_pos.x + offset_x), int(coin_pos.y + y_bob), 8, self.COLOR_COIN)

        # Draw end flag
        flag_draw_rect = self.end_flag.move(offset_x, 0)
        pygame.draw.rect(self.screen, self.COLOR_FLAG, flag_draw_rect)

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0] + offset_x), int(p['pos'][1]))
            radius = int(p['radius'])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, p['color'])

        # Draw player
        if self.jump_squash > 0:
            self.jump_squash -= 1

        squash_factor = self.jump_squash / 20.0

        player_draw_rect = self.player_rect.copy()
        player_draw_rect.w = int(self.player_rect.w * (1 + squash_factor))
        player_draw_rect.h = int(self.player_rect.h * (1 - squash_factor))
        player_draw_rect.center = self.player_rect.center

        player_draw_rect.move_ip(offset_x, 0)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_draw_rect, border_radius=4)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_sec = max(0, self.time_remaining // 30)
        time_text = self.font_ui.render(f"TIME: {time_sec}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        # Level
        level_text = self.font_level.render(f"LEVEL {self.current_level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH // 2 - level_text.get_width() // 2, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.current_level,
            "time_remaining": self.time_remaining,
        }

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


if __name__ == '__main__':
    # This block allows you to play the game with keyboard controls
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Use a persistent display window
    pygame.display.set_caption("Platformer Environment")
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    terminated = False
    total_reward = 0

    # Game loop
    running = True
    while running:
        # Action defaults
        movement = 0  # no-op

        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Keyboard controls
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        if keys[pygame.K_UP]:
            movement = 1

        if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
            running = False

        if keys[pygame.K_r]:
            obs, info = env.reset()
            total_reward = 0
            terminated = False

        action = [movement, 0, 0]  # space/shift are not used

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Control frame rate
        env.clock.tick(30)

    env.close()