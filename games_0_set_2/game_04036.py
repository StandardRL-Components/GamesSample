
# Generated: 2025-08-28T01:11:54.227808
# Source Brief: brief_04036.md
# Brief Index: 4036

        
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
        "Controls: Use arrow keys (↑↓←→) to move and evade the zombies."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down survival game. Evade the zombie horde for 100 seconds to escape the arena."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 3000  # 100 seconds at 30 FPS

        # Arena
        self.ARENA_CENTER = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.ARENA_RADIUS = 180

        # Player
        self.PLAYER_RADIUS = 8
        self.PLAYER_SPEED = 3.0

        # Zombies
        self.NUM_ZOMBIES = 20
        self.ZOMBIE_SIZE = 12
        self.ZOMBIE_SPEED = 1.0

        # Colors
        self.COLOR_BG = (25, 25, 40)
        self.COLOR_ARENA_FILL = (40, 40, 60)
        self.COLOR_ARENA_LINE = (80, 80, 110)
        self.COLOR_PLAYER = (0, 255, 127) # Spring Green
        self.COLOR_PLAYER_GLOW = (0, 255, 127, 40)
        self.COLOR_ZOMBIE = (220, 20, 60) # Crimson
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_PARTICLE = (255, 255, 255)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 32)
        
        # Game state variables
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.zombies = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.rng = None
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        
        self.player_pos = self.ARENA_CENTER.copy()
        
        self.particles = []
        self.zombies = []
        for _ in range(self.NUM_ZOMBIES):
            self.zombies.append(self._spawn_zombie())
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is already over, do nothing
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Update game logic
        self._update_player(movement)
        self._update_zombies()
        self._update_particles()
        
        # Check for termination conditions
        collision = self._check_collisions()
        time_up = self.steps >= self.MAX_STEPS
        
        terminated = collision or time_up
        if terminated:
            self.game_over = True
            self.victory = time_up and not collision

        reward = self._calculate_reward(terminated, collision)
        self.score += reward
        self.steps += 1
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_player(self, movement):
        move_vector = np.zeros(2, dtype=np.float32)
        if movement == 1:  # Up
            move_vector[1] = -1
        elif movement == 2:  # Down
            move_vector[1] = 1
        elif movement == 3:  # Left
            move_vector[0] = -1
        elif movement == 4:  # Right
            move_vector[0] = 1

        if np.linalg.norm(move_vector) > 0:
            move_vector = move_vector / np.linalg.norm(move_vector)
        
        self.player_pos += move_vector * self.PLAYER_SPEED

        # Boundary check to keep player inside arena
        dist_from_center = np.linalg.norm(self.player_pos - self.ARENA_CENTER)
        if dist_from_center > self.ARENA_RADIUS - self.PLAYER_RADIUS:
            direction = (self.player_pos - self.ARENA_CENTER) / dist_from_center
            self.player_pos = self.ARENA_CENTER + direction * (self.ARENA_RADIUS - self.PLAYER_RADIUS)
    
    def _update_zombies(self):
        for zombie in self.zombies:
            direction = self.player_pos - zombie['pos']
            dist = np.linalg.norm(direction)
            if dist > 1: # Avoid division by zero
                direction /= dist
            
            zombie['pos'] += direction * self.ZOMBIE_SPEED

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _spawn_zombie(self):
        # Spawn in a random location within the arena, but not too close to the center
        angle = self.rng.uniform(0, 2 * math.pi)
        # Spawn between 50% and 100% of the arena radius
        radius = self.rng.uniform(self.ARENA_RADIUS * 0.5, self.ARENA_RADIUS)
        pos = self.ARENA_CENTER + np.array([math.cos(angle), math.sin(angle)]) * radius
        
        # Create spawn particles
        # sfx: zombie_spawn.wav
        for _ in range(15):
            p_angle = self.rng.uniform(0, 2 * math.pi)
            p_speed = self.rng.uniform(0.5, 2.0)
            p_life = self.rng.integers(10, 20)
            self.particles.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(p_angle), math.sin(p_angle)]) * p_speed,
                'life': p_life,
                'max_life': p_life
            })
            
        return {'pos': pos}

    def _check_collisions(self):
        # sfx: player_hit.wav
        player_x, player_y = self.player_pos
        for zombie in self.zombies:
            zombie_x, zombie_y = zombie['pos']
            
            # Circle-AABB collision
            closest_x = max(zombie_x - self.ZOMBIE_SIZE/2, min(player_x, zombie_x + self.ZOMBIE_SIZE/2))
            closest_y = max(zombie_y - self.ZOMBIE_SIZE/2, min(player_y, zombie_y + self.ZOMBIE_SIZE/2))
            
            distance_sq = (player_x - closest_x)**2 + (player_y - closest_y)**2
            if distance_sq < self.PLAYER_RADIUS**2:
                return True
        return False

    def _calculate_reward(self, terminated, collision):
        if not terminated:
            return 0.1 # Survival reward
        if collision:
            return -100.0 # Penalty for dying
        else: # Survived the full time
            # sfx: victory_fanfare.wav
            return 100.0

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw arena
        pygame.gfxdraw.filled_circle(
            self.screen, int(self.ARENA_CENTER[0]), int(self.ARENA_CENTER[1]),
            self.ARENA_RADIUS, self.COLOR_ARENA_FILL
        )
        pygame.gfxdraw.aacircle(
            self.screen, int(self.ARENA_CENTER[0]), int(self.ARENA_CENTER[1]),
            self.ARENA_RADIUS, self.COLOR_ARENA_LINE
        )

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            size = int(2 * (p['life'] / p['max_life']))
            if size > 0:
                pygame.draw.circle(
                    self.screen,
                    (self.COLOR_PARTICLE[0], self.COLOR_PARTICLE[1], self.COLOR_PARTICLE[2], alpha),
                    (int(p['pos'][0]), int(p['pos'][1])), size
                )

        # Draw zombies
        for zombie in self.zombies:
            zombie_rect = pygame.Rect(
                zombie['pos'][0] - self.ZOMBIE_SIZE / 2,
                zombie['pos'][1] - self.ZOMBIE_SIZE / 2,
                self.ZOMBIE_SIZE, self.ZOMBIE_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, zombie_rect)

        # Draw player
        player_int_pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        
        # Glow effect
        glow_surf = pygame.Surface((self.PLAYER_RADIUS * 4, self.PLAYER_RADIUS * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2), self.PLAYER_RADIUS * 1.8)
        self.screen.blit(glow_surf, (player_int_pos[0] - self.PLAYER_RADIUS * 2, player_int_pos[1] - self.PLAYER_RADIUS * 2))

        pygame.gfxdraw.filled_circle(self.screen, player_int_pos[0], player_int_pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_int_pos[0], player_int_pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_ui(self):
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = f"TIME: {time_left:.1f}"
        text_surface = self.font_small.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))
        
        # Game Over / Victory message
        if self.game_over:
            if self.victory:
                message = "VICTORY!"
            else:
                message = "GAME OVER"
            
            message_surface = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = message_surface.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(message_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        }
    
    def close(self):
        pygame.font.quit()
        pygame.quit()