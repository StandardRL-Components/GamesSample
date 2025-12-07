
# Generated: 2025-08-27T18:18:47.493125
# Source Brief: brief_01790.md
# Brief Index: 1790

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys for isometric movement. "
        "↑ (Up-Left), ↓ (Down-Right), ← (Down-Left), → (Up-Right)."
    )

    game_description = (
        "Navigate a procedurally generated cavern, collecting glowing crystals to score points. "
        "Avoid the red patrolling enemies. High-value crystals spawn near enemies, offering a risky reward."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Critical Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 50
        self.INITIAL_HEALTH = 100
        self.PLAYER_SPEED = 0.4
        self.PLAYER_COLLISION_RADIUS = 1.0
        self.ENEMY_COLLISION_RADIUS = 1.2
        self.CRYSTAL_COLLISION_RADIUS = 1.0
        
        # --- Visuals ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255, 50)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_ENEMY_GLOW = (255, 80, 80, 40)
        self.COLOR_CRYSTAL_REG = (100, 200, 255)
        self.COLOR_CRYSTAL_HIGH = (255, 220, 100)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_HEART = (255, 80, 80)
        
        # --- World Parameters ---
        self.WORLD_SIZE = 40  # Logical grid size
        self.TILE_WIDTH = 32
        self.TILE_HEIGHT = 16

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24)
        self.font_msg = pygame.font.SysFont("Verdana", 48, bold=True)

        # --- State Variables ---
        self.np_random = None
        self.player_pos = None
        self.player_health = 0
        self.enemies = []
        self.crystals = []
        self.particles = []
        self.background_elements = []
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_enemy_collision_step = -100 # Cooldown for damage

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = self.INITIAL_HEALTH
        self.last_enemy_collision_step = -100
        
        self.player_pos = self.np_random.uniform(self.WORLD_SIZE * 0.4, self.WORLD_SIZE * 0.6, size=2)

        self.enemies.clear()
        self.crystals.clear()
        self.particles.clear()
        self.background_elements.clear()

        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        # Generate background elements
        for _ in range(80):
            pos = self.np_random.uniform(0, self.WORLD_SIZE, size=2)
            radius = self.np_random.uniform(1.5, 4)
            color_shade = self.np_random.integers(25, 45)
            color = (color_shade, color_shade + 5, color_shade + 15)
            self.background_elements.append({'pos': pos, 'radius': radius, 'color': color})

        # Generate enemies and their paths
        num_enemies = 5
        for _ in range(num_enemies):
            path_len = self.np_random.integers(3, 6)
            path = [self.np_random.uniform(0, self.WORLD_SIZE, size=2) for _ in range(path_len)]
            enemy = {
                'path': path,
                'path_index': 0,
                'pos': np.copy(path[0]),
                'speed': self.np_random.uniform(0.05, 0.1)
            }
            self.enemies.append(enemy)
            
            # Place high-value crystals near enemy paths
            for point in path:
                if self.np_random.random() < 0.7:
                    offset = self.np_random.uniform(-3, 3, size=2)
                    crystal_pos = np.clip(point + offset, 0, self.WORLD_SIZE - 1)
                    self.crystals.append({'pos': crystal_pos, 'value': 5, 'is_high': True})

        # Generate regular crystals
        num_crystals = 60
        while len(self.crystals) < num_crystals + num_enemies * 2:
            pos = self.np_random.uniform(0, self.WORLD_SIZE, size=2)
            # Ensure not too close to player start or other crystals
            if np.linalg.norm(pos - self.player_pos) > 8 and all(np.linalg.norm(pos - c['pos']) > 2 for c in self.crystals):
                self.crystals.append({'pos': pos, 'value': 1, 'is_high': False})

    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            # Store old distances for reward calculation
            old_dist_crystal = self._get_dist_to_nearest_crystal()
            old_dist_enemy = self._get_dist_to_nearest_enemy()

            self._move_player(action)
            self._move_enemies()

            reward += self._check_crystal_collisions()
            reward += self._check_enemy_collisions()

            # Continuous reward shaping
            new_dist_crystal = self._get_dist_to_nearest_crystal()
            new_dist_enemy = self._get_dist_to_nearest_enemy()

            if new_dist_crystal < old_dist_crystal:
                reward += 0.1
            if new_dist_enemy < old_dist_enemy:
                reward -= 0.2

            self.steps += 1
            terminated = self._check_termination()
            if terminated:
                self.game_over = True
                if self.score >= self.WIN_SCORE:
                    reward += 100
                if self.player_health <= 0:
                    reward -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_player(self, action):
        movement = action[0]
        direction = np.array([0.0, 0.0])
        if movement == 1: direction = np.array([-1.0, -1.0]) # Up-Left
        elif movement == 2: direction = np.array([1.0, 1.0])   # Down-Right
        elif movement == 3: direction = np.array([-1.0, 1.0])  # Down-Left
        elif movement == 4: direction = np.array([1.0, -1.0])   # Up-Right
        
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        
        self.player_pos += direction * self.PLAYER_SPEED
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WORLD_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.WORLD_SIZE)

    def _move_enemies(self):
        for enemy in self.enemies:
            target_point = enemy['path'][enemy['path_index']]
            direction = target_point - enemy['pos']
            dist = np.linalg.norm(direction)

            if dist < 0.2:
                enemy['path_index'] = (enemy['path_index'] + 1) % len(enemy['path'])
            else:
                enemy['pos'] += (direction / dist) * enemy['speed']

    def _check_crystal_collisions(self):
        reward = 0
        collected_indices = []
        for i, crystal in enumerate(self.crystals):
            if np.linalg.norm(self.player_pos - crystal['pos']) < self.CRYSTAL_COLLISION_RADIUS:
                reward += crystal['value']
                self.score += crystal['value']
                collected_indices.append(i)
                color = self.COLOR_CRYSTAL_HIGH if crystal['is_high'] else self.COLOR_CRYSTAL_REG
                # Sound: Crystal collect sound
                self._create_particles(crystal['pos'], color)

        for i in sorted(collected_indices, reverse=True):
            del self.crystals[i]
        return reward

    def _check_enemy_collisions(self):
        # Cooldown to prevent instant death from continuous contact
        if self.steps < self.last_enemy_collision_step + 30:
            return 0
        
        for enemy in self.enemies:
            if np.linalg.norm(self.player_pos - enemy['pos']) < self.ENEMY_COLLISION_RADIUS:
                self.player_health -= 25
                self.last_enemy_collision_step = self.steps
                # Sound: Player damage sound
                self._create_particles(self.player_pos, self.COLOR_ENEMY, 20)
                return -5
        return 0

    def _get_dist_to_nearest_crystal(self):
        if not self.crystals: return 0
        return min(np.linalg.norm(self.player_pos - c['pos']) for c in self.crystals)

    def _get_dist_to_nearest_enemy(self):
        if not self.enemies: return float('inf')
        return min(np.linalg.norm(self.player_pos - e['pos']) for e in self.enemies)

    def _check_termination(self):
        return self.player_health <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS

    def _iso_to_cart(self, iso_pos):
        player_cart_x = (self.player_pos[0] - self.player_pos[1]) * (self.TILE_WIDTH / 2)
        player_cart_y = (self.player_pos[0] + self.player_pos[1]) * (self.TILE_HEIGHT / 2)
        iso_cart_x = (iso_pos[0] - iso_pos[1]) * (self.TILE_WIDTH / 2)
        iso_cart_y = (iso_pos[0] + iso_pos[1]) * (self.TILE_HEIGHT / 2)
        
        screen_x = self.WIDTH / 2 + iso_cart_x - player_cart_x
        screen_y = self.HEIGHT / 2.5 + iso_cart_y - player_cart_y
        return int(screen_x), int(screen_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._update_and_render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        render_list = []
        for element in self.background_elements:
            render_list.append({'pos': element['pos'], 'type': 'bg', 'data': element})
        for crystal in self.crystals:
            render_list.append({'pos': crystal['pos'], 'type': 'crystal', 'data': crystal})
        for enemy in self.enemies:
            render_list.append({'pos': enemy['pos'], 'type': 'enemy', 'data': enemy})
        render_list.append({'pos': self.player_pos, 'type': 'player', 'data': {}})

        render_list.sort(key=lambda item: (item['pos'][1], item['pos'][0]))

        for item in render_list:
            screen_pos = self._iso_to_cart(item['pos'])
            if item['type'] == 'bg':
                pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], int(item['data']['radius']), item['data']['color'])
            elif item['type'] == 'crystal':
                self._render_crystal(screen_pos, item['data'])
            elif item['type'] == 'enemy':
                self._render_enemy(screen_pos)
            elif item['type'] == 'player':
                self._render_player(screen_pos)

    def _render_crystal(self, screen_pos, crystal):
        pulse = (math.sin(self.steps * 0.1 + crystal['pos'][0] * 5) + 1) / 2
        size = 5 + pulse * 2
        color = self.COLOR_CRYSTAL_HIGH if crystal['is_high'] else self.COLOR_CRYSTAL_REG
        if crystal['is_high']: size += 2

        points = [
            (screen_pos[0], screen_pos[1] - size), (screen_pos[0] + size * 0.7, screen_pos[1]),
            (screen_pos[0], screen_pos[1] + size), (screen_pos[0] - size * 0.7, screen_pos[1])
        ]
        
        glow_size = size * (2.0 + pulse * 0.5)
        glow_alpha = 70 + pulse * 40
        glow_points = [
            (screen_pos[0], screen_pos[1] - glow_size), (screen_pos[0] + glow_size * 0.7, screen_pos[1]),
            (screen_pos[0], screen_pos[1] + glow_size), (screen_pos[0] - glow_size * 0.7, screen_pos[1])
        ]
        pygame.gfxdraw.filled_polygon(self.screen, glow_points, (*color, int(glow_alpha)))
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _render_enemy(self, screen_pos):
        size = 10
        points = [
            (screen_pos[0], screen_pos[1] - size * 0.8), (screen_pos[0] + size, screen_pos[1]),
            (screen_pos[0], screen_pos[1] + size * 0.8), (screen_pos[0] - size, screen_pos[1])
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY_GLOW)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)

    def _render_player(self, screen_pos):
        size = 12
        points = [
            (screen_pos[0], screen_pos[1] - size * 0.8), (screen_pos[0] + size, screen_pos[1]),
            (screen_pos[0], screen_pos[1] + size * 0.8), (screen_pos[0] - size, screen_pos[1])
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)

    def _create_particles(self, iso_pos, color, count=15):
        screen_pos = self._iso_to_cart(iso_pos)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed - 1]
            self.particles.append({
                'pos': list(screen_pos), 'vel': velocity,
                'life': self.np_random.integers(15, 30), 'color': color
            })

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / 30))
                size = max(1, int(p['life'] / 10))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, (*p['color'], alpha))

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        health_text = self.font_ui.render(f"{max(0, int(self.player_health))}", True, self.COLOR_UI_TEXT)
        heart_pos = (self.WIDTH - health_text.get_width() - 35, 22)
        self.screen.blit(health_text, (self.WIDTH - health_text.get_width() - 10, 10))
        
        heart_points = [
            (heart_pos[0], heart_pos[1]-5), (heart_pos[0]+5, heart_pos[1]-10), (heart_pos[0]+10, heart_pos[1]-5),
            (heart_pos[0], heart_pos[1]+5),
            (heart_pos[0]-10, heart_pos[1]-5), (heart_pos[0]-5, heart_pos[1]-10)
        ]
        pygame.gfxdraw.aapolygon(self.screen, heart_points, self.COLOR_HEART)
        pygame.gfxdraw.filled_polygon(self.screen, heart_points, self.COLOR_HEART)

        if self.game_over:
            msg_text = "VICTORY!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = (150, 255, 150) if self.score >= self.WIN_SCORE else (255, 100, 100)
            rendered_msg = self.font_msg.render(msg_text, True, color)
            msg_rect = rendered_msg.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            
            bg_rect = msg_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((20, 25, 40, 180))
            self.screen.blit(s, bg_rect)
            self.screen.blit(rendered_msg, msg_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "health": self.player_health}

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")