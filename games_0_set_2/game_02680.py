
# Generated: 2025-08-27T21:07:18.129705
# Source Brief: brief_02680.md
# Brief Index: 2680

        
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
        "Controls: ←→ to move, ↑ or Space to jump. Collect coins and reach the flag!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Leap across procedurally generated platforms, collecting coins and striving for the flag in this side-scrolling platformer."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Colors
    COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
    COLOR_BG_BOTTOM = (34, 139, 34)  # Forest Green
    COLOR_PLATFORM = (60, 179, 113)  # Medium Sea Green
    COLOR_PLATFORM_OUTLINE = (46, 139, 87) # Sea Green
    COLOR_PLAYER = (255, 69, 0)  # OrangeRed
    COLOR_PLAYER_OUTLINE = (255, 255, 255)
    COLOR_COIN = (255, 215, 0)  # Gold
    COLOR_FLAGPOLE = (139, 69, 19) # SaddleBrown
    COLOR_FLAG = (220, 20, 60) # Crimson
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_OUTLINE = (0, 0, 0)

    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Physics
    GRAVITY = 0.4
    PLAYER_ACCEL = 0.8
    FRICTION = 0.85
    MAX_VX = 6.0
    JUMP_POWER = -9.0
    PLAYER_SIZE = 20

    # World Generation
    WORLD_LENGTH = 12000 # pixels
    MAX_STEPS = 2000

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
        self.font_ui = pygame.font.SysFont("Arial", 24, bold=True)
        
        # Initialize state variables
        self.player_x = 0.0
        self.player_y = 0.0
        self.player_vx = 0.0
        self.player_vy = 0.0
        self.is_grounded = False
        self.player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        self.camera_x = 0.0
        self.camera_y = 0.0
        
        self.platforms = []
        self.coins = []
        self.particles = []
        
        self.flag_pole_rect = pygame.Rect(0, 0, 0, 0)
        self.flag_rect = pygame.Rect(0, 0, 0, 0)

        self.steps = 0
        self.score = 0
        self.last_player_x = 0.0
        self.game_over = False

        self.render_mode = render_mode
        self.reset()
        
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_x = 100.0
        self.player_y = 200.0
        self.player_vx = 0.0
        self.player_vy = 0.0
        self.is_grounded = False
        self.last_player_x = self.player_x

        self.camera_x = 0.0
        self.camera_y = 100.0

        self.platforms = []
        self.coins = []
        self.particles = []

        self._generate_initial_world()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _generate_initial_world(self):
        # Starting platform
        start_platform = pygame.Rect(20, 250, 200, self.SCREEN_HEIGHT)
        self.platforms.append(start_platform)
        
        # Procedural generation
        current_x = start_platform.right
        last_y = start_platform.y
        
        while current_x < self.WORLD_LENGTH:
            difficulty_mod = self.steps / 200.0
            
            gap = self.np_random.integers(60, 100) + difficulty_mod * 5
            current_x += gap
            
            width = self.np_random.integers(80, 200)
            
            height_diff = self.np_random.integers(-60, 60) + self.np_random.choice([-1, 1]) * difficulty_mod * 2
            new_y = np.clip(last_y + height_diff, 150, self.SCREEN_HEIGHT - 50)
            
            platform_rect = pygame.Rect(current_x, new_y, width, self.SCREEN_HEIGHT)
            self.platforms.append(platform_rect)

            # Add coins on platforms
            if self.np_random.random() < 0.6:
                num_coins = self.np_random.integers(1, 4)
                for i in range(num_coins):
                    coin_x = platform_rect.x + (i + 1) * (platform_rect.width / (num_coins + 1))
                    coin_y = platform_rect.y - 30
                    self.coins.append(pygame.Rect(int(coin_x), int(coin_y), 12, 12))

            current_x += width
            last_y = new_y
            
        # Flag at the end
        flag_x = self.WORLD_LENGTH - 100
        flag_base_y = 0
        for p in self.platforms:
            if p.left < flag_x < p.right:
                flag_base_y = p.top
                break
        self.flag_pole_rect = pygame.Rect(flag_x, flag_base_y - 100, 10, 100)
        self.flag_rect = pygame.Rect(flag_x - 40, flag_base_y - 90, 40, 30)

    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        # shift_held is unused

        reward = 0
        terminated = self.game_over

        # --- Handle Input ---
        if movement == 3:  # Left
            self.player_vx -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_vx += self.PLAYER_ACCEL
        
        self.player_vx *= self.FRICTION
        self.player_vx = np.clip(self.player_vx, -self.MAX_VX, self.MAX_VX)

        is_jump_action = (movement == 1 or space_held)
        if is_jump_action and self.is_grounded:
            self.player_vy = self.JUMP_POWER
            self.is_grounded = False
            # SFX: jump.wav
            self._create_particles(self.player_rect.midbottom, 10, (150, 150, 150))
            
        # --- Physics Update ---
        self.player_vy += self.GRAVITY
        
        self.player_x += self.player_vx
        self.player_rect.centerx = int(self.player_x)
        
        for plat in self.platforms:
            if self.player_rect.colliderect(plat):
                if self.player_vx > 0:
                    self.player_rect.right = plat.left
                elif self.player_vx < 0:
                    self.player_rect.left = plat.right
                self.player_vx = 0
                self.player_x = float(self.player_rect.centerx)
        
        self.player_y += self.player_vy
        self.player_rect.centery = int(self.player_y)
        
        self.is_grounded = False
        for plat in self.platforms:
            if self.player_rect.colliderect(plat) and self.player_vy >= 0:
                # Check if player was above the platform in the previous frame
                if (self.player_rect.centery - self.player_vy) <= plat.top:
                    self.player_rect.bottom = plat.top
                    self.player_vy = 0
                    self.is_grounded = True
                    # SFX: land.wav
                    break # Stop checking after first ground collision
        self.player_y = float(self.player_rect.centery)

        # --- Rewards & Game Logic ---
        progress = self.player_x - self.last_player_x
        reward += progress * 0.1 # Per brief: reward for moving towards flag
        self.last_player_x = self.player_x

        collected_coins = []
        for coin in self.coins:
            if self.player_rect.colliderect(coin):
                collected_coins.append(coin)
                self.score += 1
                reward += 1.0 # Per brief: +1 for coin
                # SFX: coin.wav
                self._create_particles(coin.center, 15, self.COLOR_COIN)
        self.coins = [c for c in self.coins if c not in collected_coins]

        # --- Update Camera & Particles ---
        self.camera_x = self.player_x - self.SCREEN_WIDTH / 2
        target_cam_y = self.player_y - self.SCREEN_HEIGHT * 0.6
        self.camera_y += (target_cam_y - self.camera_y) * 0.1
        self._update_particles()
        
        # --- Termination Conditions ---
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        if self.player_y > self.SCREEN_HEIGHT + 100:
            terminated = True
            reward = -100.0 # Per brief
            # SFX: fall.wav
        
        if self.player_rect.colliderect(self.flag_rect):
            terminated = True
            reward += 100.0 # Per brief
            self.score += 100
            # SFX: win.wav
            self._create_particles(self.player_rect.center, 50, self.COLOR_FLAG)
        
        self.game_over = terminated
            
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _create_particles(self, pos, count, color):
        for _ in range(count):
            self.particles.append({
                'pos': list(pos),
                'vel': [self.np_random.uniform(-2, 2), self.np_random.uniform(-3, 1)],
                'lifespan': self.np_random.integers(15, 30),
                'color': color,
                'radius': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1
            p['lifespan'] -= 1
            p['radius'] -= 0.1
            if p['lifespan'] > 0 and p['radius'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        # Draw gradient background
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = tuple(
                int(self.COLOR_BG_TOP[i] * (1 - ratio) + self.COLOR_BG_BOTTOM[i] * ratio)
                for i in range(3)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))
        
        self._render_game()
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cam_x, cam_y = int(self.camera_x), int(self.camera_y)

        # Draw platforms
        for plat in self.platforms:
            if plat.right > cam_x and plat.left < cam_x + self.SCREEN_WIDTH:
                render_rect = plat.move(-cam_x, -cam_y)
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, render_rect)
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM_OUTLINE, render_rect, 2)

        # Draw coins
        for coin in self.coins:
            if coin.right > cam_x and coin.left < cam_x + self.SCREEN_WIDTH:
                render_pos = (coin.centerx - cam_x, coin.centery - cam_y)
                spin_factor = abs(math.sin((self.steps + coin.x) * 0.1))
                radius = coin.width / 2
                pygame.gfxdraw.filled_ellipse(self.screen, render_pos[0], render_pos[1], int(max(1, radius * spin_factor)), int(radius), self.COLOR_COIN)
                pygame.gfxdraw.aaellipse(self.screen, render_pos[0], render_pos[1], int(max(1, radius * spin_factor)), int(radius), self.COLOR_COIN)

        # Draw flag
        if self.flag_pole_rect.right > cam_x and self.flag_pole_rect.left < cam_x + self.SCREEN_WIDTH:
            pygame.draw.rect(self.screen, self.COLOR_FLAGPOLE, self.flag_pole_rect.move(-cam_x, -cam_y))
            pygame.draw.polygon(self.screen, self.COLOR_FLAG, [
                (self.flag_rect.left - cam_x, self.flag_rect.top - cam_y),
                (self.flag_rect.left - cam_x, self.flag_rect.bottom - cam_y),
                (self.flag_rect.right - cam_x, self.flag_rect.centery - cam_y)
            ])

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0] - cam_x), int(p['pos'][1] - cam_y))
            pygame.draw.circle(self.screen, p['color'], pos, int(p['radius']))

        # Draw player
        player_screen_rect = self.player_rect.move(-cam_x, -cam_y)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_screen_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_screen_rect, 2, border_radius=3)

    def _render_text_with_outline(self, font, text, pos, color, outline_color):
        text_surface = font.render(text, True, color)
        outline_surface = font.render(text, True, outline_color)
        offsets = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        for dx, dy in offsets:
            self.screen.blit(outline_surface, (pos[0] + dx, pos[1] + dy))
        self.screen.blit(text_surface, pos)

    def _render_ui(self):
        score_text = f"SCORE: {self.score}"
        self._render_text_with_outline(self.font_ui, score_text, (20, 20), self.COLOR_TEXT, self.COLOR_TEXT_OUTLINE)
        
        steps_text = f"STEPS: {self.steps} / {self.MAX_STEPS}"
        steps_size = self.font_ui.size(steps_text)
        self._render_text_with_outline(self.font_ui, steps_text, (self.SCREEN_WIDTH - steps_size[0] - 20, 20), self.COLOR_TEXT, self.COLOR_TEXT_OUTLINE)

        # Progress bar
        progress_ratio = np.clip(self.player_x / self.WORLD_LENGTH, 0, 1)
        bar_width = self.SCREEN_WIDTH / 2
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        bar_height = 10
        pygame.draw.rect(self.screen, self.COLOR_TEXT_OUTLINE, (bar_x, 25, bar_width, bar_height), 1, border_radius=2)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (bar_x + 2, 27, max(0, (bar_width - 4) * progress_ratio), bar_height - 4), border_radius=2)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment for human play
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    # env.validate_implementation() # Uncomment to run self-check
    
    obs, info = env.reset()
    terminated = False
    
    pygame.display.set_caption(env.game_description)
    real_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    movement, space, shift = 0, 0, 0

    print(env.user_guide)

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: shift = 1
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    movement, space, shift = 0, 0, 0
            
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT] and movement == {pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4}.get(event.key):
                    movement = 0
                if event.key == pygame.K_SPACE: space = 0
                if event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: shift = 0

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)

    env.close()
    print(f"Game Over. Final Score: {info['score']}, Steps: {info['steps']}")