
# Generated: 2025-08-28T03:24:19.012930
# Source Brief: brief_02005.md
# Brief Index: 2005

        
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
    """
    A fast-paced arcade gem-grabbing game.

    The player controls a yellow square and must collect all cyan gems while
    avoiding red triangular enemies. Enemies patrol in fixed circular paths.
    The episode ends if the player collides with an enemy, collects all gems,
    or the time limit (1000 steps) is reached.

    Rewards are structured to encourage efficient gem collection and risky plays.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move. Collect all the gems and avoid the red enemies!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game where you collect gems while dodging patrolling enemies. "
        "Get bonus points for risky grabs near enemies."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GAME_AREA_PADDING = 20

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
        
        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_GLOW = (255, 255, 150)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_GEM = (0, 255, 255)
        self.COLOR_UI = (255, 255, 255)
        self.COLOR_BORDER = (100, 100, 120)

        # Game constants
        self.PLAYER_SIZE = 16
        self.PLAYER_SPEED = 4.0
        self.GEM_SIZE = 6
        self.ENEMY_SIZE = 20
        self.NUM_GEMS = 50
        self.NUM_ENEMIES = 5
        self.BASE_ENEMY_SPEED = 1.0
        self.RISK_BONUS_DISTANCE = 75
        self.MAX_STEPS = 1000

        # State variables (initialized in reset)
        self.player_pos = None
        self.player_rect = None
        self.gems = None
        self.enemies = None
        self.particles = None
        self.score = 0
        self.steps = 0
        self.level = 1
        self.game_over = False
        self.np_random = None
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Initialize all game state
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.level = 1

        self._setup_level()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _setup_level(self):
        # Player
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        self.player_rect.center = self.player_pos

        # Gems
        self.gems = []
        min_x = self.GAME_AREA_PADDING + self.GEM_SIZE
        max_x = self.WIDTH - self.GAME_AREA_PADDING - self.GEM_SIZE
        min_y = self.GAME_AREA_PADDING + self.GEM_SIZE
        max_y = self.HEIGHT - self.GAME_AREA_PADDING - self.GEM_SIZE
        
        while len(self.gems) < self.NUM_GEMS:
            pos_x = self.np_random.uniform(min_x, max_x)
            pos_y = self.np_random.uniform(min_y, max_y)
            new_gem = pygame.Rect(pos_x - self.GEM_SIZE, pos_y - self.GEM_SIZE, self.GEM_SIZE * 2, self.GEM_SIZE * 2)
            
            if not new_gem.colliderect(self.player_rect.inflate(50, 50)) and not any(g.colliderect(new_gem) for g in self.gems):
                self.gems.append(new_gem)

        # Enemies
        self.enemies = []
        enemy_speed = self.BASE_ENEMY_SPEED + (self.level - 1) * 0.2
        for _ in range(self.NUM_ENEMIES):
            center = self.np_random.uniform(
                [min_x, min_y], [max_x, max_y], size=2
            )
            radius = self.np_random.uniform(50, 150)
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed_multiplier = self.np_random.uniform(0.8, 1.2)
            
            enemy_pos_x = center[0] + radius * math.cos(angle)
            enemy_pos_y = center[1] + radius * math.sin(angle)

            self.enemies.append({
                "center": center,
                "radius": radius,
                "angle": angle,
                "speed": (enemy_speed * speed_multiplier) / max(1, radius),
                "pos": np.array([enemy_pos_x, enemy_pos_y], dtype=np.float32),
                "rect": pygame.Rect(0, 0, self.ENEMY_SIZE, self.ENEMY_SIZE)
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held = action[1] == 1 # Not used
        # shift_held = action[2] == 1 # Not used

        reward = -0.02 # Time penalty
        terminated = False

        dist_to_gem_before, _ = self._get_nearest_entity_dist(self.player_pos, self.gems)
        dist_to_enemy_before, _ = self._get_nearest_entity_dist(self.player_pos, [e['rect'] for e in self.enemies])
        
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3: self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: self.player_pos[0] += self.PLAYER_SPEED
        
        self.player_pos[0] = np.clip(self.player_pos[0], self.GAME_AREA_PADDING + self.PLAYER_SIZE/2, self.WIDTH - self.GAME_AREA_PADDING - self.PLAYER_SIZE/2)
        self.player_pos[1] = np.clip(self.player_pos[1], self.GAME_AREA_PADDING + self.PLAYER_SIZE/2, self.HEIGHT - self.GAME_AREA_PADDING - self.PLAYER_SIZE/2)
        self.player_rect.center = (int(self.player_pos[0]), int(self.player_pos[1]))

        for enemy in self.enemies:
            enemy['angle'] += enemy['speed']
            enemy['pos'][0] = enemy['center'][0] + enemy['radius'] * math.cos(enemy['angle'])
            enemy['pos'][1] = enemy['center'][1] + enemy['radius'] * math.sin(enemy['angle'])
            enemy['rect'].center = (int(enemy['pos'][0]), int(enemy['pos'][1]))

        self._update_particles()

        dist_to_gem_after, _ = self._get_nearest_entity_dist(self.player_pos, self.gems)
        dist_to_enemy_after, _ = self._get_nearest_entity_dist(self.player_pos, [e['rect'] for e in self.enemies])

        if dist_to_gem_after is not None and dist_to_gem_before is not None and dist_to_gem_after < dist_to_gem_before:
            reward += 0.1
        if dist_to_enemy_after is not None and dist_to_enemy_before is not None and dist_to_enemy_after < dist_to_enemy_before:
            reward -= 0.2

        collected_gems = [gem for gem in self.gems if self.player_rect.colliderect(gem)]
        for gem in collected_gems:
            self.gems.remove(gem)
            self.score += 10
            reward += 10
            # # Sound: gem_collect.wav
            if dist_to_enemy_after is not None and dist_to_enemy_after < self.RISK_BONUS_DISTANCE:
                reward += 5
                self.score += 5 # Bonus score
                self._create_particles(gem.center, self.COLOR_PLAYER, 10, 2.0)
            self._create_particles(gem.center, self.COLOR_GEM, 15)
        
        if any(self.player_rect.colliderect(e['rect']) for e in self.enemies):
            reward = -100
            terminated = True
            self.game_over = True
            # # Sound: player_hit.wav
            self._create_particles(self.player_rect.center, self.COLOR_ENEMY, 50, 4.0)
        
        if not terminated:
            if not self.gems:
                reward += 100
                terminated = True
                self.game_over = True
                # # Sound: level_complete.wav
            
            self.steps += 1
            if self.steps >= self.MAX_STEPS:
                terminated = True
                self.game_over = True

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
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
        border_rect = pygame.Rect(self.GAME_AREA_PADDING, self.GAME_AREA_PADDING, self.WIDTH - 2*self.GAME_AREA_PADDING, self.HEIGHT - 2*self.GAME_AREA_PADDING)
        pygame.draw.rect(self.screen, self.COLOR_BORDER, border_rect, 1)

        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), max(1, int(p['life'] / 8)))

        for enemy in self.enemies:
            angle = enemy['angle'] + math.pi/2 # rotate to point "up" relative to circle path
            points = [
                (enemy['pos'][0] + math.sin(angle) * self.ENEMY_SIZE * 0.8, enemy['pos'][1] - math.cos(angle) * self.ENEMY_SIZE * 0.8),
                (enemy['pos'][0] + math.sin(angle + 2.5) * self.ENEMY_SIZE * 0.6, enemy['pos'][1] - math.cos(angle + 2.5) * self.ENEMY_SIZE * 0.6),
                (enemy['pos'][0] + math.sin(angle - 2.5) * self.ENEMY_SIZE * 0.6, enemy['pos'][1] - math.cos(angle - 2.5) * self.ENEMY_SIZE * 0.6),
            ]
            int_points = [(int(p[0]), int(p[1])) for p in points]
            pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_ENEMY)

        for gem in self.gems:
            center = (int(gem.centerx), int(gem.centery))
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], self.GEM_SIZE, self.COLOR_GEM)
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], self.GEM_SIZE, self.COLOR_GEM)

        if not (self.game_over and len(self.gems) > 0):
            glow_size = self.PLAYER_SIZE * 0.8 + (math.sin(self.steps * 0.1) + 1) * 2
            glow_surface = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, (*self.COLOR_PLAYER_GLOW, 50), (glow_size, glow_size), glow_size)
            self.screen.blit(glow_surface, (self.player_rect.centerx - glow_size, self.player_rect.centery - glow_size), special_flags=pygame.BLEND_RGBA_ADD)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
        
        if self.game_over:
            msg = "LEVEL COMPLETE!" if not self.gems else "GAME OVER"
            color = self.COLOR_GEM if not self.gems else self.COLOR_ENEMY
            end_text = self.font.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return { "score": self.score, "steps": self.steps, "gems_remaining": len(self.gems) }

    def _create_particles(self, pos, color, count, speed_scale=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1.0, 3.0) * speed_scale
            self.particles.append({
                "pos": np.array(pos, dtype=np.float32),
                "vel": np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                "life": self.np_random.integers(15, 30),
                "color": color
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 1]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['life'] -= 1

    def _get_distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def _get_nearest_entity_dist(self, pos, entities):
        if not entities: return None, None
        dists = [self._get_distance(pos, e.center) for e in entities]
        min_idx = np.argmin(dists)
        return dists[min_idx], entities[min_idx]

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")