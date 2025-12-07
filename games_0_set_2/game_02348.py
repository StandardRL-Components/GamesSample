
# Generated: 2025-08-27T20:06:10.356982
# Source Brief: brief_02348.md
# Brief Index: 2348

        
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
        "Controls: Arrow keys to move. Survive the horde for 60 seconds."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down arcade survival game. Dodge the ever-growing zombie horde for as long as you can. The longer you survive, the higher your score!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Constants
        self.FPS = 60
        self.MAX_STEPS = self.FPS * 60  # 60 seconds

        # Colors
        self.COLOR_BG = (16, 16, 16)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 50)
        self.COLOR_ZOMBIE = (255, 50, 50)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_BAR = (0, 204, 0)
        self.COLOR_HEALTH_BAR_BG = (136, 0, 0)
        self.COLOR_HIT_FLASH = (255, 0, 0)

        # Game parameters
        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 3.5
        self.ZOMBIE_SIZE = 18
        self.ZOMBIE_SPEED = 2.0
        self.MAX_HEALTH = 5
        self.INITIAL_SPAWN_INTERVAL = self.FPS * 5 # 5 seconds
        self.MIN_SPAWN_INTERVAL = self.FPS * 0.5 # 0.5 seconds

        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Initialize state variables
        self.player_pos = None
        self.health = 0
        self.zombies = []
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.zombie_spawn_timer = 0
        self.hit_flash_timer = 0

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(
            self.WIDTH / 2, self.HEIGHT / 2
        )
        self.health = self.MAX_HEALTH
        self.zombies = []
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.zombie_spawn_timer = self.INITIAL_SPAWN_INTERVAL
        self.hit_flash_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        self._handle_player_movement(movement)
        
        self._update_zombies()
        self._spawn_zombies()
        self._update_particles()
        
        if self.hit_flash_timer > 0:
            self.hit_flash_timer -= 1

        self.steps += 1
        
        # Calculate rewards
        reward = self._handle_collisions()
        reward += 0.01  # Survival reward
        if movement == 0:
            reward -= 0.2 # Penalty for no-op

        # Check termination conditions
        terminated = self.health <= 0 or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if self.health > 0:  # Survived the full time
                reward += 100.0
                # Sound: victory_jingle.wav

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_movement(self, movement):
        if movement == 1: # Up
            self.player_pos.y -= self.PLAYER_SPEED
        elif movement == 2: # Down
            self.player_pos.y += self.PLAYER_SPEED
        elif movement == 3: # Left
            self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos.x += self.PLAYER_SPEED
        
        # Clamp player position to screen bounds
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE/2, self.WIDTH - self.PLAYER_SIZE/2)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE/2, self.HEIGHT - self.PLAYER_SIZE/2)

    def _spawn_zombies(self):
        self.zombie_spawn_timer -= 1
        if self.zombie_spawn_timer <= 0:
            # Spawn a new zombie
            # Sound: zombie_spawn.wav
            side = self.np_random.integers(0, 4)
            if side == 0: # Top
                pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -self.ZOMBIE_SIZE)
            elif side == 1: # Bottom
                pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ZOMBIE_SIZE)
            elif side == 2: # Left
                pos = pygame.Vector2(-self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT))
            else: # Right
                pos = pygame.Vector2(self.WIDTH + self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT))

            # Zombie moves towards player's position at spawn time
            direction = (self.player_pos - pos).normalize()
            velocity = direction * self.ZOMBIE_SPEED
            
            self.zombies.append({'pos': pos, 'vel': velocity})
            
            # Decrease spawn interval to increase difficulty
            progress = self.steps / self.MAX_STEPS
            current_interval = self.INITIAL_SPAWN_INTERVAL - (self.INITIAL_SPAWN_INTERVAL - self.MIN_SPAWN_INTERVAL) * progress
            self.zombie_spawn_timer = current_interval

    def _update_zombies(self):
        for z in self.zombies:
            z['pos'] += z['vel']
        
        # Remove zombies that are far off-screen
        self.zombies = [z for z in self.zombies if -100 < z['pos'].x < self.WIDTH + 100 and -100 < z['pos'].y < self.HEIGHT + 100]

    def _handle_collisions(self):
        hit_reward = 0.0
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE/2, self.player_pos.y - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        remaining_zombies = []
        for z in self.zombies:
            zombie_rect = pygame.Rect(z['pos'].x - self.ZOMBIE_SIZE/2, z['pos'].y - self.ZOMBIE_SIZE/2, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
            if player_rect.colliderect(zombie_rect):
                self.health -= 1
                hit_reward -= 10.0
                self.hit_flash_timer = 5 # Flash for 5 frames
                self._create_particles(z['pos'])
                # Sound: player_hit.wav
                if self.health <= 0:
                    # Sound: game_over.wav
                    pass
            else:
                remaining_zombies.append(z)
        
        self.zombies = remaining_zombies
        return hit_reward

    def _create_particles(self, position):
        # Sound: explosion.wav
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifetime = self.np_random.integers(20, 40)
            self.particles.append({'pos': position.copy(), 'vel': velocity, 'life': lifetime, 'max_life': lifetime})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (self.COLOR_ZOMBIE[0], self.COLOR_ZOMBIE[1], self.COLOR_ZOMBIE[2], alpha)
            size = int(self.ZOMBIE_SIZE/2 * (p['life'] / p['max_life']))
            if size > 0:
                pygame.draw.circle(self.screen, color, (int(p['pos'].x), int(p['pos'].y)), size)

        # Render zombies
        for z in self.zombies:
            z_rect = pygame.Rect(int(z['pos'].x - self.ZOMBIE_SIZE/2), int(z['pos'].y - self.ZOMBIE_SIZE/2), self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, z_rect)
            
        # Render player glow
        glow_size = int(self.PLAYER_SIZE * 2.5)
        glow_surface = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, self.COLOR_PLAYER_GLOW, (glow_size//2, glow_size//2), glow_size//2)
        self.screen.blit(glow_surface, (int(self.player_pos.x - glow_size/2), int(self.player_pos.y - glow_size/2)))

        # Render player
        player_rect = pygame.Rect(int(self.player_pos.x - self.PLAYER_SIZE/2), int(self.player_pos.y - self.PLAYER_SIZE/2), self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        
        # Render hit flash
        if self.hit_flash_timer > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            alpha = int(150 * (self.hit_flash_timer / 5))
            flash_surface.fill((self.COLOR_HIT_FLASH[0], self.COLOR_HIT_FLASH[1], self.COLOR_HIT_FLASH[2], alpha))
            self.screen.blit(flash_surface, (0, 0))

    def _render_ui(self):
        # Health Bar
        health_bar_width = 150
        health_bar_height = 20
        health_pct = max(0, self.health / self.MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, health_bar_width, health_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(health_bar_width * health_pct), health_bar_height))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_small.render(f"Time: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))
        
        # Score
        score_text = self.font_small.render(f"Score: {self.score:.2f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, self.HEIGHT - score_text.get_height() - 10))
        
        # Game Over Text
        if self.game_over:
            if self.health <= 0:
                end_text_str = "GAME OVER"
            else:
                end_text_str = "YOU SURVIVED!"
            
            end_text = self.font_large.render(end_text_str, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.health,
        }

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()