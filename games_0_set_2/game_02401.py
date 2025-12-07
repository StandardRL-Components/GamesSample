
# Generated: 2025-08-28T04:46:53.829510
# Source Brief: brief_02401.md
# Brief Index: 2401

        
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
        "Controls: ←→ to move, ↑ to jump. Collect 50 coins and reach the green flag."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Leap across procedurally generated platforms, collecting coins to reach the end flag in this side-scrolling arcade platformer."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and rendering
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_end = pygame.font.SysFont("monospace", 60, bold=True)

        # Game constants
        self.GRAVITY = 0.6
        self.JUMP_STRENGTH = -12
        self.MOVE_SPEED = 7
        self.FRICTION = 0.85
        self.MAX_VEL_Y = 15
        self.PLAYER_SIZE = (20, 20)
        self.WORLD_LENGTH = 8000
        self.MAX_STEPS = 1000
        self.COIN_WIN_CONDITION = 50

        # Colors
        self.COLOR_BG_TOP = (48, 25, 52)
        self.COLOR_BG_BOTTOM = (20, 12, 28)
        self.COLOR_PLAYER = (255, 64, 64)
        self.COLOR_PLAYER_GLOW = (255, 128, 128)
        self.PLATFORM_COLORS = [(80, 144, 255), (100, 160, 255), (60, 120, 230)]
        self.COLOR_COIN = (255, 215, 0)
        self.COLOR_FLAG = (0, 200, 0)
        self.COLOR_TEXT = (255, 255, 255)
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0 # Represents coin count
        self.game_over = False
        
        self.player_rect = pygame.Rect(100, self.HEIGHT / 2, self.PLAYER_SIZE[0], self.PLAYER_SIZE[1])
        self.player_vel = [0.0, 0.0]
        self.is_grounded = True
        
        self.camera_x = 0.0
        self.max_reached_x = self.player_rect.x
        
        self.platforms = []
        self.coins = []
        self.particles = []
        
        # Create starting platform
        start_platform = pygame.Rect(0, self.HEIGHT - 40, 300, 40)
        self.platforms.append(start_platform)
        
        self._generate_world()
        
        flag_y = self._find_y_at_x(self.WORLD_LENGTH) - 50
        self.end_flag_rect = pygame.Rect(self.WORLD_LENGTH, flag_y, 10, 50)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        # 1. Handle player input
        if movement == 3: # Left
            self.player_vel[0] = -self.MOVE_SPEED
        elif movement == 4: # Right
            self.player_vel[0] = self.MOVE_SPEED
        else: # No horizontal input, apply friction
            self.player_vel[0] *= self.FRICTION
            if abs(self.player_vel[0]) < 0.1: self.player_vel[0] = 0

        if movement == 1 and self.is_grounded:
            self.player_vel[1] = self.JUMP_STRENGTH
            self.is_grounded = False
            # sfx: jump

        # 2. Update physics
        self.player_vel[1] += self.GRAVITY
        self.player_vel[1] = min(self.player_vel[1], self.MAX_VEL_Y)
        
        self.player_rect.x += self.player_vel[0]
        self.player_rect.y += self.player_vel[1]
        
        self.is_grounded = False
        
        # 3. Handle collisions
        for plat in self.platforms:
            if self.player_rect.colliderect(plat) and self.player_vel[1] >= 0:
                if self.player_rect.bottom - self.player_vel[1] <= plat.top + 1:
                    self.player_rect.bottom = plat.top
                    if self.player_vel[1] > 1: # Only trigger landing effects if falling
                        self._create_particles(self.player_rect.midbottom, 5, (200, 200, 200))
                        # sfx: land
                    self.player_vel[1] = 0
                    self.is_grounded = True
                    break
        
        # 4. Calculate reward and handle collections
        reward = 0.0
        
        progress = self.player_rect.x - self.max_reached_x
        if progress > 0:
            reward += progress * 0.1
            self.max_reached_x = self.player_rect.x
            
        collected_coins = [c for c in self.coins if self.player_rect.colliderect(c)]
        if collected_coins:
            self.coins = [c for c in self.coins if c not in collected_coins]
            num_collected = len(collected_coins)
            self.score += num_collected
            reward += num_collected * 1.0
            for coin in collected_coins:
                self._create_particles(coin.center, 10, self.COLOR_COIN)
                # sfx: coin collect
        
        # 5. Update game state
        self.steps += 1
        self.camera_x = max(self.camera_x, self.player_rect.centerx - self.WIDTH / 3)
        self._update_particles()
        self._cull_objects()
        
        # 6. Check for termination
        terminated = False
        if self.player_rect.top > self.HEIGHT:
            terminated = True # Fell off screen
            # sfx: fall/lose
        elif self.steps >= self.MAX_STEPS:
            terminated = True # Out of time
        elif self.player_rect.colliderect(self.end_flag_rect):
            terminated = True # Reached flag
            if self.score >= self.COIN_WIN_CONDITION:
                reward = 100.0 # Win
                # sfx: win
            else:
                reward = -100.0 # Lose
                # sfx: lose
        
        if terminated:
            self.game_over = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = tuple(int(c1 * (1 - interp) + c2 * interp) for c1, c2 in zip(self.COLOR_BG_TOP, self.COLOR_BG_BOTTOM))
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Draw platforms
        for plat in self.platforms:
            screen_rect = plat.move(-self.camera_x, 0)
            if screen_rect.right > 0 and screen_rect.left < self.WIDTH:
                color_index = int(plat.x / 200) % len(self.PLATFORM_COLORS)
                pygame.draw.rect(self.screen, self.PLATFORM_COLORS[color_index], screen_rect, border_radius=3)
                highlight_rect = pygame.Rect(screen_rect.left, screen_rect.top, screen_rect.width, 4)
                pygame.draw.rect(self.screen, (255,255,255, 50), highlight_rect, border_radius=3)

        # Draw coins
        for coin in self.coins:
            screen_rect = coin.move(-self.camera_x, 0)
            if screen_rect.right > 0 and screen_rect.left < self.WIDTH:
                scale = math.sin(self.steps * 0.2) * 0.5 + 0.5
                width = max(2, int(coin.width * scale))
                anim_rect = pygame.Rect(screen_rect.centerx - width // 2, screen_rect.y, width, coin.height)
                pygame.draw.ellipse(self.screen, self.COLOR_COIN, anim_rect)
                pygame.draw.ellipse(self.screen, (255, 255, 150), anim_rect, width=2)

        # Draw end flag
        screen_flag_rect = self.end_flag_rect.move(-self.camera_x, 0)
        if screen_flag_rect.right > 0 and screen_flag_rect.left < self.WIDTH:
            pole_rect = pygame.Rect(int(screen_flag_rect.left), int(screen_flag_rect.top), 5, int(screen_flag_rect.height))
            pygame.draw.rect(self.screen, (192, 192, 192), pole_rect)
            wave = math.sin(self.steps * 0.1) * 5
            p1 = (pole_rect.right, pole_rect.top)
            p2 = (pole_rect.right, pole_rect.top + 25)
            p3 = (pole_rect.right + 40 + wave, pole_rect.top + 12.5)
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_FLAG)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_FLAG)
            
        # Draw particles
        for p in self.particles:
            screen_pos = (int(p['pos'][0] - self.camera_x), int(p['pos'][1]))
            alpha = int(255 * (p['lifespan'] / 30.0))
            color_with_alpha = p['color'] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], max(0, int(p['radius'])), color_with_alpha)

        # Draw player
        screen_player_rect = self.player_rect.move(-self.camera_x, 0)
        glow_rect = screen_player_rect.inflate(10, 10)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_PLAYER_GLOW + (50,), s.get_rect(), border_radius=8)
        self.screen.blit(s, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, screen_player_rect, border_radius=4)
        
    def _render_ui(self):
        coin_text = f"COINS: {self.score}"
        text_surface = self.font_ui.render(coin_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))

        if self.game_over:
            win = self.player_rect.colliderect(self.end_flag_rect) and self.score >= self.COIN_WIN_CONDITION
            end_text = "YOU WIN!" if win else "GAME OVER"
            end_color = self.COLOR_FLAG if win else self.COLOR_PLAYER
            end_surf = self.font_end.render(end_text, True, end_color)
            end_rect = end_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_surf, end_rect)

    def _generate_world(self):
        initial_gap = 40
        max_gap = 100
        while self.platforms[-1].right < self.WORLD_LENGTH + self.WIDTH:
            last_platform = self.platforms[-1]
            # Difficulty scales with number of platforms generated so far
            gap_increase = (len(self.platforms) // 10) * 2
            current_gap = initial_gap + gap_increase
            
            x = last_platform.right + self.np_random.uniform(current_gap, min(max_gap, current_gap + 20))
            y_change = self.np_random.uniform(-80, 80)
            y = np.clip(last_platform.centery + y_change, 150, self.HEIGHT - 40)
            width = self.np_random.uniform(60, 200)
            
            new_platform = pygame.Rect(x, y, width, 20)
            self.platforms.append(new_platform)
            
            if self.np_random.random() < 0.5:
                self.coins.append(pygame.Rect(new_platform.centerx - 10, new_platform.top - 30, 20, 20))

    def _find_y_at_x(self, x_pos):
        closest_platform = min(self.platforms, key=lambda p: abs(p.centerx - x_pos), default=None)
        return closest_platform.top if closest_platform else self.HEIGHT - 100

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            radius = self.np_random.uniform(2, 5)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'radius': radius, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['radius'] -= 0.1
        self.particles = [p for p in self.particles if p['lifespan'] > 0 and p['radius'] > 0]

    def _cull_objects(self):
        cull_line = self.camera_x - self.WIDTH
        self.platforms = [p for p in self.platforms if p.right > cull_line]
        self.coins = [c for c in self.coins if c.right > cull_line]

    def close(self):
        pygame.font.quit()
        pygame.quit()