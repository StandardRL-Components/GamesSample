
# Generated: 2025-08-28T06:59:40.679890
# Source Brief: brief_03103.md
# Brief Index: 3103

        
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
        "Controls: ←→ to select jump direction. Press Space to jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hop between procedurally generated platforms to reach the top. The platforms are constantly moving up!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and game dimensions
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
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG_TOP = (20, 30, 80)
        self.COLOR_BG_BOTTOM = (60, 120, 200)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_GLOW = (150, 255, 150, 50) # RGBA
        self.COLOR_PLATFORM = (240, 240, 255)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_PARTICLE = (220, 220, 220)
        
        # Game constants
        self.PLAYER_SIZE = 20
        self.GRAVITY = 0.6
        self.JUMP_POWER = -13
        self.JUMP_HORIZ_SPEED = 6
        self.PLATFORM_HEIGHT = 15
        self.PLATFORM_WIDTH_RANGE = (80, 150)
        self.NUM_PLATFORMS = 15
        self.PLATFORM_V_SPACING = 120
        self.MAX_EPISODE_STEPS = 10000
        self.INITIAL_PLATFORM_SPEED = 1.0

        # Initialize state variables that are reset
        self.player_pos = [0, 0]
        self.player_vel = [0, 0]
        self.on_ground = False
        self.last_space_held = False
        self.platforms = []
        self.particles = []
        self.platform_speed = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.highest_platform_idx = 0
        self.current_platform_idx = 0
        
        self.np_random = None
        
        # Initialize state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.platform_speed = self.INITIAL_PLATFORM_SPEED
        self.highest_platform_idx = 0
        self.current_platform_idx = 0
        self.last_space_held = False
        self.particles.clear()

        # Generate platforms
        self.platforms.clear()
        start_platform_width = self.WIDTH // 3
        start_platform = pygame.Rect(
            (self.WIDTH - start_platform_width) / 2,
            self.HEIGHT - 50,
            start_platform_width,
            self.PLATFORM_HEIGHT,
        )
        self.platforms.append(start_platform)

        last_x = start_platform.centerx
        for i in range(1, self.NUM_PLATFORMS):
            width = self.np_random.integers(self.PLATFORM_WIDTH_RANGE[0], self.PLATFORM_WIDTH_RANGE[1])
            max_offset = self.JUMP_HORIZ_SPEED * abs(self.JUMP_POWER / self.GRAVITY) * 0.6
            
            x_offset = self.np_random.uniform(-max_offset, max_offset)
            x = last_x + x_offset
            x = np.clip(x, width / 2, self.WIDTH - width / 2)
            
            y = self.platforms[-1].top - self.PLATFORM_V_SPACING + self.np_random.integers(-20, 20)
            
            self.platforms.append(pygame.Rect(x - width / 2, y, width, self.PLATFORM_HEIGHT))
            last_x = x
        
        # Make final platform special
        final_plat = self.platforms[-1]
        final_plat.width = self.WIDTH
        final_plat.x = 0

        # Initialize player state
        self.player_pos = [start_platform.centerx, start_platform.top - self.PLAYER_SIZE]
        self.player_vel = [0, 0]
        self.on_ground = True
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # Unpack factorized action
        movement = action[0]  # 3=left, 4=right
        space_held = action[1] == 1  # Boolean
        
        # --- Handle Input ---
        if self.on_ground and space_held and not self.last_space_held:
            self.on_ground = False
            self.player_vel[1] = self.JUMP_POWER
            # // Sound effect: Jump
            if movement == 3:  # Left
                self.player_vel[0] = -self.JUMP_HORIZ_SPEED
            elif movement == 4:  # Right
                self.player_vel[0] = self.JUMP_HORIZ_SPEED
            else: # Straight up
                self.player_vel[0] = 0
        
        self.last_space_held = space_held

        # --- Physics Update ---
        if not self.on_ground:
            self.player_vel[1] += self.GRAVITY
        
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]

        if self.on_ground:
            self.player_vel[0] *= 0.8
        
        if self.player_pos[0] < 0:
            self.player_pos[0] = 0
            self.player_vel[0] *= -0.5
        elif self.player_pos[0] > self.WIDTH - self.PLAYER_SIZE:
            self.player_pos[0] = self.WIDTH - self.PLAYER_SIZE
            self.player_vel[0] *= -0.5

        for plat in self.platforms:
            plat.y += self.platform_speed

        if self.on_ground:
            self.player_pos[1] += self.platform_speed
            reward += 0.1

        # --- Collision Detection ---
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        landed = False
        if self.player_vel[1] > 0:
            for i, plat in enumerate(self.platforms):
                if player_rect.colliderect(plat) and abs(player_rect.bottom - plat.top) < self.player_vel[1] + self.platform_speed + 2:
                    self.player_pos[1] = plat.top - self.PLAYER_SIZE
                    self.player_vel[1] = 0
                    self.on_ground = True
                    landed = True
                    # // Sound effect: Land
                    self._create_particles(player_rect.midbottom)
                    
                    if i > self.highest_platform_idx:
                        reward += 10
                        self.highest_platform_idx = i
                    elif i < self.current_platform_idx:
                        reward -= 1
                    
                    self.current_platform_idx = i
                    break
        
        if landed and self.current_platform_idx == len(self.platforms) - 1:
            self.game_over = True
            reward += 100
        
        # --- Update Difficulty ---
        if self.steps > 0 and self.steps % 500 == 0:
            self.platform_speed += 0.05
        
        # --- Termination Check ---
        terminated = self.game_over
        if self.player_pos[1] > self.HEIGHT:
            terminated = True
            self.game_over = True
            reward -= 50
        
        if self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _create_particles(self, pos):
        for _ in range(10):
            vel = [self.np_random.uniform(-2, 2), self.np_random.uniform(-3, -0.5)]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'max_life': life})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_game(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = tuple(
                int(self.COLOR_BG_TOP[i] * (1 - interp) + self.COLOR_BG_BOTTOM[i] * interp)
                for i in range(3)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        for i, plat in enumerate(self.platforms):
            color = self.COLOR_PLATFORM
            if i == len(self.platforms) - 1:
                color = (255, 215, 0)
            pygame.draw.rect(self.screen, color, plat, border_radius=3)
            border_color = tuple(c * 0.8 for c in color)
            pygame.draw.rect(self.screen, border_color, plat, width=2, border_radius=3)

        self._update_particles()
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            size = max(0, int(5 * (p['life'] / p['max_life'])))
            if size > 0:
                s = pygame.Surface((size, size), pygame.SRCALPHA)
                s.fill(self.COLOR_PARTICLE + (alpha,))
                self.screen.blit(s, (int(p['pos'][0] - size/2), int(p['pos'][1] - size/2)))
        
        glow_size = self.PLAYER_SIZE * 1.8
        glow_surface = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, self.COLOR_PLAYER_GLOW, (0, 0, glow_size, glow_size), border_radius=int(glow_size/3))
        self.screen.blit(glow_surface, (int(self.player_pos[0] - (glow_size - self.PLAYER_SIZE) / 2), 
                                        int(self.player_pos[1] - (glow_size - self.PLAYER_SIZE) / 2)))

        player_rect = pygame.Rect(int(self.player_pos[0]), int(self.player_pos[1]), self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        player_border = tuple(c * 0.8 for c in self.COLOR_PLAYER)
        pygame.draw.rect(self.screen, player_border, player_rect, width=2, border_radius=3)
        
    def _render_ui(self):
        score_text = self.font.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        height_text = self.small_font.render(f"Height: {self.highest_platform_idx}/{self.NUM_PLATFORMS-1}", True, self.COLOR_TEXT)
        self.screen.blit(height_text, (10, 40))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WON!" if self.current_platform_idx == len(self.platforms) - 1 else "GAME OVER"
            end_text = self.font.render(end_text_str, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG_TOP)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": self.highest_platform_idx,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Hopper")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30)
        
    env.close()