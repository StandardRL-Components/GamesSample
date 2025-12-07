
# Generated: 2025-08-27T18:50:58.682889
# Source Brief: brief_01974.md
# Brief Index: 1974

        
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
    A fast-paced arcade game where the player collects gems while dodging enemies.
    The goal is to collect 100 gems to win, or survive as long as possible to maximize score.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Dodge the red triangles and collect the cyan gems."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Grab gems, dodge enemies, and maximize your score in this fast-paced top-down arcade game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 5
        self.GEM_SIZE = 10
        self.ENEMY_SIZE = 24
        self.NUM_GEMS_ON_SCREEN = 15
        self.NUM_ENEMIES = 5
        self.MAX_STEPS = 1000
        self.WIN_CONDITION_GEMS = 100
        self.ENEMY_PROXIMITY_THRESHOLD = 80

        # Colors
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_GLOW = (255, 255, 0, 60)
        self.COLOR_GEM = (0, 255, 255)
        self.COLOR_GEM_GLOW = (200, 255, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_GLOW = (255, 100, 100)
        self.COLOR_UI = (255, 255, 255)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        # Game state variables (initialized in reset)
        self.player_pos = None
        self.enemies = None
        self.gems = None
        self.particles = None
        self.score = None
        self.steps = None
        self.gems_collected_count = None
        self.terminated = None
        self.prev_dist_to_gem = None
        self.prev_dist_to_enemy = None

        self.reset()
        # self.validate_implementation() # Uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)

        self.gems = [self._spawn_gem() for _ in range(self.NUM_GEMS_ON_SCREEN)]
        self.enemies = [self._spawn_enemy() for _ in range(self.NUM_ENEMIES)]
        self.particles = []

        self.score = 0
        self.steps = 0
        self.gems_collected_count = 0
        self.terminated = False

        self.prev_dist_to_gem = self._get_closest_dist(self.player_pos, self.gems)
        self.prev_dist_to_enemy = self._get_closest_dist(self.player_pos, [e['pos'] for e in self.enemies])

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        # 1. Update game state based on action and time
        self._handle_action(action)
        self._update_enemies()
        self._update_particles()

        # 2. Calculate rewards and handle events
        # Distance-based shaping rewards
        current_dist_to_gem = self._get_closest_dist(self.player_pos, self.gems)
        if current_dist_to_gem < self.prev_dist_to_gem:
            reward += 1.0  # +1 for moving closer to a gem
        elif current_dist_to_gem > self.prev_dist_to_gem:
            reward -= 2.0  # -2 for moving away from a gem
        self.prev_dist_to_gem = current_dist_to_gem

        current_dist_to_enemy = self._get_closest_dist(self.player_pos, [e['pos'] for e in self.enemies])
        if current_dist_to_enemy < self.prev_dist_to_enemy:
            reward -= 0.1 # -0.1 for moving closer to an enemy
        self.prev_dist_to_enemy = current_dist_to_enemy
        
        # Proximity bonus
        if current_dist_to_enemy < self.ENEMY_PROXIMITY_THRESHOLD:
            reward += 5.0

        # Gem collection
        collected_indices = []
        for i, gem_pos in enumerate(self.gems):
            if self._check_collision(self.player_pos, self.PLAYER_SIZE/2, gem_pos, self.GEM_SIZE):
                collected_indices.append(i)
                self.score += 10
                reward += 10
                self.gems_collected_count += 1
                self._create_particles(gem_pos, self.COLOR_GEM)
                # sfx: gem collect sound

        for i in sorted(collected_indices, reverse=True):
            self.gems.pop(i)
            self.gems.append(self._spawn_gem())

        # 3. Check for termination conditions
        self.steps += 1
        
        # Enemy collision
        for enemy in self.enemies:
            if self._check_collision(self.player_pos, self.PLAYER_SIZE/2, enemy['pos'], self.ENEMY_SIZE/2):
                self.terminated = True
                reward = -100  # -100 for colliding with an enemy
                # sfx: explosion sound
                break
        
        # Win/loss conditions
        if not self.terminated:
            if self.gems_collected_count >= self.WIN_CONDITION_GEMS:
                self.terminated = True
                reward = 100  # +100 for collecting 100 gems
            elif self.steps >= self.MAX_STEPS:
                self.terminated = True

        return (
            self._get_observation(),
            reward,
            self.terminated,
            False,
            self._get_info(),
        )

    def _handle_action(self, action):
        movement = action[0]
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED  # Up
        elif movement == 2: self.player_pos[1] += self.PLAYER_SPEED  # Down
        elif movement == 3: self.player_pos[0] -= self.PLAYER_SPEED  # Left
        elif movement == 4: self.player_pos[0] += self.PLAYER_SPEED  # Right

        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE / 2, self.WIDTH - self.PLAYER_SIZE / 2)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE / 2, self.HEIGHT - self.PLAYER_SIZE / 2)

    def _update_enemies(self):
        for enemy in self.enemies:
            enemy['angle'] += enemy['speed']
            offset_x = math.cos(enemy['angle']) * enemy['radius']
            offset_y = math.sin(enemy['angle']) * enemy['radius']
            enemy['pos'][0] = enemy['center'][0] + offset_x
            enemy['pos'][1] = enemy['center'][1] + offset_y

    def _spawn_gem(self):
        return np.array([
            self.np_random.uniform(self.GEM_SIZE, self.WIDTH - self.GEM_SIZE),
            self.np_random.uniform(self.GEM_SIZE, self.HEIGHT - self.GEM_SIZE)
        ], dtype=np.float32)

    def _spawn_enemy(self):
        padding = 100
        center = np.array([
            self.np_random.uniform(padding, self.WIDTH - padding),
            self.np_random.uniform(padding, self.HEIGHT - padding)
        ], dtype=np.float32)
        radius = self.np_random.uniform(40, 80)
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(0.01, 0.03) * self.np_random.choice([-1, 1])
        pos = center + np.array([math.cos(angle) * radius, math.sin(angle) * radius])
        return {'pos': pos, 'center': center, 'radius': radius, 'angle': angle, 'speed': speed}

    def _check_collision(self, pos1, r1, pos2, r2):
        return np.linalg.norm(pos1 - pos2) < (r1 + r2)

    def _get_closest_dist(self, pos, object_list):
        if not object_list:
            return float('inf')
        object_positions = np.array(object_list)
        distances = np.linalg.norm(object_positions - pos, axis=1)
        return np.min(distances)

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            velocity = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)
            self.particles.append({
                'pos': pos.copy(),
                'vel': velocity,
                'life': self.np_random.integers(10, 20),
                'max_life': 20
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['life'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw enemies
        for enemy in self.enemies:
            p = enemy['pos']
            s = self.ENEMY_SIZE / 2
            points = [
                (p[0], p[1] - s * 1.15),
                (p[0] - s, p[1] + s * 0.85),
                (p[0] + s, p[1] + s * 0.85)
            ]
            int_points = [(int(x), int(y)) for x, y in points]
            pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_ENEMY)

        # Draw gems
        for gem_pos in self.gems:
            x, y = int(gem_pos[0]), int(gem_pos[1])
            r = self.GEM_SIZE
            pygame.gfxdraw.aacircle(self.screen, x, y, r, self.COLOR_GEM_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, x, y, r - 2, self.COLOR_GEM)

        # Draw particles
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            alpha = max(0, min(255, int(255 * life_ratio)))
            color = self.COLOR_GEM + (alpha,)
            size = int(max(1, self.GEM_SIZE / 2 * life_ratio))
            
            # Use a temporary surface for alpha blending
            temp_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (int(p['pos'][0] - size), int(p['pos'][1] - size)))

        # Draw player
        px, py = self.player_pos
        ps = self.PLAYER_SIZE
        glow_surf = pygame.Surface((ps * 2.5, ps * 2.5), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (ps * 1.25, ps * 1.25), ps * 1.25)
        self.screen.blit(glow_surf, (int(px - ps * 1.25), int(py - ps * 1.25)))
        player_rect = pygame.Rect(int(px - ps / 2), int(py - ps / 2), ps, ps)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)

    def _render_ui(self):
        score_surf = self.font.render(f"Score: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_surf, (10, 10))
        
        gem_surf = self.font.render(f"Gems: {self.gems_collected_count} / {self.WIN_CONDITION_GEMS}", True, self.COLOR_UI)
        self.screen.blit(gem_surf, (10, 45))
        
        steps_surf = self.font.render(f"Steps: {self.steps} / {self.MAX_STEPS}", True, self.COLOR_UI)
        self.screen.blit(steps_surf, (self.WIDTH - steps_surf.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_collected": self.gems_collected_count,
            "player_pos": self.player_pos,
            "distance_to_gem": self.prev_dist_to_gem,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """ Call this at the end of __init__ to verify implementation. """
        print("Running implementation validation...")
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
        assert "score" in info
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        assert "steps" in info

        print("✓ Implementation validated successfully")