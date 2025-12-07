
# Generated: 2025-08-27T14:32:12.314513
# Source Brief: brief_00712.md
# Brief Index: 712

        
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

    user_guide = (
        "Controls: Use arrow keys to aim. Press space to fire. Hold shift to upgrade your ship (costs score)."
    )

    game_description = (
        "A turn-based arcade shooter. Destroy all asteroids in the grid. "
        "Destroyed asteroids create dangerous debris. Upgrade your ship to increase projectile range."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 10
        self.CELL_SIZE = 40
        self.GRID_X_OFFSET = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_Y_OFFSET = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        
        self.MAX_STEPS = 1000
        self.INITIAL_ASTEROIDS = 20
        self.UPGRADE_COST = 300
        self.MAX_UPGRADES = 2

        # Colors
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_GRID = (30, 35, 60)
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_SHIP = [(50, 205, 50), (60, 180, 255), (180, 100, 255)] # Green, Blue, Purple
        self.COLOR_ASTEROID = (255, 80, 80)
        self.COLOR_DEBRIS = (255, 200, 0)
        self.COLOR_PROJECTILE = (255, 255, 255)
        self.COLOR_AIM = (255, 255, 255, 100)
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        self.ship_pos = None
        self.aim_direction = None
        self.upgrade_level = 0
        self.projectile_range = 0
        self.asteroids = []
        self.debris = []
        self.projectiles = []
        self.particles = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.ship_pos = (self.GRID_WIDTH // 2 -1, self.GRID_HEIGHT - 1)
        self.aim_direction = (0, -1) # Start aiming up
        self.upgrade_level = 0
        self.projectile_range = 4

        self.asteroids = []
        self.debris = []
        self.projectiles = []
        self.particles = []

        # Generate asteroids
        possible_spawns = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        ship_area = set()
        for i in range(-2, 3):
            for j in range(-2, 3):
                ship_area.add((self.ship_pos[0] + i, self.ship_pos[1] + j))
        
        possible_spawns = [p for p in possible_spawns if p not in ship_area]
        
        spawn_indices = self.np_random.choice(len(possible_spawns), self.INITIAL_ASTEROIDS, replace=False)
        self.asteroids = [possible_spawns[i] for i in spawn_indices]

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        shot_fired = False

        # 1. Process player action (Priority: Fire > Upgrade > Aim)
        if space_held:
            # Fire projectile
            start_pos = (self.ship_pos[0] + 0.5, self.ship_pos[1] + 0.5)
            self.projectiles.append({
                "pos": list(start_pos),
                "vel": self.aim_direction,
                "dist": 0
            })
            shot_fired = True
            reward -= 0.1 # Penalty for firing, offset by hit reward
            # sfx: player_shoot.wav
        elif shift_held:
            # Upgrade ship
            if self.upgrade_level < self.MAX_UPGRADES and self.score >= self.UPGRADE_COST:
                self.score -= self.UPGRADE_COST
                self.upgrade_level += 1
                self.projectile_range += 1
                reward += 2 # Reward for making a good decision
                # sfx: upgrade.wav
        elif movement > 0:
            # Change aim
            dirs = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # up, down, left, right
            self.aim_direction = dirs[movement]

        # 2. Update game state
        self._update_projectiles()
        hit_reward, score_gain = self._check_projectile_collisions()
        reward += hit_reward
        self.score += score_gain

        self._move_entities(self.asteroids)
        self._move_entities(self.debris)
        
        self._update_particles()
        
        # 3. Check for termination conditions
        terminated = False
        # Debris collision with ship
        if self.ship_pos in self.debris:
            terminated = True
            reward = -5 # Per brief
            self.game_over = True
            self._create_explosion(self.ship_pos, self.COLOR_SHIP[self.upgrade_level], 50)
            # sfx: player_explosion.wav
        
        # All asteroids destroyed
        if not self.asteroids:
            terminated = True
            reward += 50
            self.game_over = True
            # sfx: victory.wav

        # Max steps reached
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True

        if terminated:
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_projectiles(self):
        for p in self.projectiles:
            p["pos"][0] += p["vel"][0] * 0.8 # Speed
            p["pos"][1] += p["vel"][1] * 0.8
            p["dist"] += 0.8

    def _check_projectile_collisions(self):
        reward, score = 0, 0
        projectiles_to_remove = []
        asteroids_to_remove = set()

        for i, p in enumerate(self.projectiles):
            p_grid_pos = (int(p["pos"][0]), int(p["pos"][1]))
            
            if p_grid_pos in self.asteroids and p_grid_pos not in asteroids_to_remove:
                reward += 11 # +1 for hit, +10 for destroy
                score += 100
                asteroids_to_remove.add(p_grid_pos)
                projectiles_to_remove.append(i)
                self.debris.append(p_grid_pos)
                self._create_explosion(p_grid_pos, self.COLOR_ASTEROID, 30)
                # sfx: asteroid_explosion.wav
            
            elif p["dist"] > self.projectile_range or not (0 <= p_grid_pos[0] < self.GRID_WIDTH and 0 <= p_grid_pos[1] < self.GRID_HEIGHT):
                projectiles_to_remove.append(i)

        if asteroids_to_remove:
            self.asteroids = [a for a in self.asteroids if a not in asteroids_to_remove]
        
        for i in sorted(projectiles_to_remove, reverse=True):
            del self.projectiles[i]
            
        return reward, score

    def _move_entities(self, entity_list):
        moves = [(0,0), (0,1), (0,-1), (1,0), (-1,0)]
        for i in range(len(entity_list)):
            dx, dy = moves[self.np_random.integers(len(moves))]
            new_pos = (
                max(0, min(self.GRID_WIDTH - 1, entity_list[i][0] + dx)),
                max(0, min(self.GRID_HEIGHT - 1, entity_list[i][1] + dy))
            )
            entity_list[i] = new_pos

    def _grid_to_pixel(self, grid_pos, center=False):
        x, y = grid_pos
        px = self.GRID_X_OFFSET + x * self.CELL_SIZE
        py = self.GRID_Y_OFFSET + y * self.CELL_SIZE
        if center:
            px += self.CELL_SIZE // 2
            py += self.CELL_SIZE // 2
        return int(px), int(py)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid()
        if not self.game_over:
            self._render_aim_indicator()
            self._render_ship()
        self._render_projectiles()
        self._render_debris()
        self._render_asteroids()
        self._render_particles()

    def _render_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            px = self.GRID_X_OFFSET + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_Y_OFFSET), (px, self.GRID_Y_OFFSET + self.GRID_HEIGHT * self.CELL_SIZE))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.GRID_Y_OFFSET + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, py), (self.GRID_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE, py))

    def _render_ship(self):
        ship_center = self._grid_to_pixel(self.ship_pos, center=True)
        color = self.COLOR_SHIP[self.upgrade_level]
        
        # Simple triangle shape
        s = self.CELL_SIZE * 0.4
        points = [
            (ship_center[0], ship_center[1] - s),
            (ship_center[0] - s, ship_center[1] + s),
            (ship_center[0] + s, ship_center[1] + s),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_aim_indicator(self):
        start_pos = self._grid_to_pixel(self.ship_pos, center=True)
        for i in range(1, self.projectile_range + 1):
            end_pos = (
                start_pos[0] + self.aim_direction[0] * i * self.CELL_SIZE,
                start_pos[1] + self.aim_direction[1] * i * self.CELL_SIZE
            )
            alpha = 150 - (i / self.projectile_range) * 120
            pygame.draw.circle(self.screen, (255, 255, 255, alpha), end_pos, 2, 1)

    def _render_asteroids(self):
        size = self.CELL_SIZE * 0.8
        offset = (self.CELL_SIZE - size) / 2
        for pos in self.asteroids:
            px, py = self._grid_to_pixel(pos)
            rect = pygame.Rect(px + offset, py + offset, size, size)
            pygame.draw.rect(self.screen, self.COLOR_ASTEROID, rect, border_radius=3)

    def _render_debris(self):
        size = self.CELL_SIZE * 0.3
        offset = (self.CELL_SIZE - size) / 2
        for pos in self.debris:
            px, py = self._grid_to_pixel(pos)
            rect = pygame.Rect(px + offset, py + offset, size, size)
            pygame.draw.rect(self.screen, self.COLOR_DEBRIS, rect)

    def _render_projectiles(self):
        for p in self.projectiles:
            px, py = self._grid_to_pixel(p["pos"])
            end_x = px - p["vel"][0] * self.CELL_SIZE * 0.5
            end_y = py - p["vel"][1] * self.CELL_SIZE * 0.5
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, (px, py), (end_x, end_y), 3)

    def _render_particles(self):
        for p in self.particles:
            p.update()
            p.draw(self.screen)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.is_alive()]
        
    def _create_explosion(self, grid_pos, color, count):
        px, py = self._grid_to_pixel(grid_pos, center=True)
        for _ in range(count):
            self.particles.append(self.Particle(px, py, color, self.np_random))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))
        
        # Asteroids remaining
        asteroids_text = self.font_main.render(f"ASTEROIDS: {len(self.asteroids)}", True, self.COLOR_TEXT)
        self.screen.blit(asteroids_text, (self.SCREEN_WIDTH - asteroids_text.get_width() - 20, 15))

        # Upgrade status
        upgrade_str = f"UPGRADE LVL: {self.upgrade_level}/{self.MAX_UPGRADES}"
        if self.upgrade_level < self.MAX_UPGRADES:
            upgrade_str += f" (Cost: {self.UPGRADE_COST})"
        upgrade_text = self.font_small.render(upgrade_str, True, self.COLOR_SHIP[self.upgrade_level])
        self.screen.blit(upgrade_text, (20, 45))

        if self.game_over:
            end_text_str = "VICTORY!" if not self.asteroids else "GAME OVER"
            end_text = self.font_main.render(end_text_str, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "asteroids_left": len(self.asteroids),
            "upgrade_level": self.upgrade_level,
        }

    def validate_implementation(self):
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

    class Particle:
        def __init__(self, x, y, color, rng):
            self.x = x
            self.y = y
            self.color = color
            self.rng = rng
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            self.vx = math.cos(angle) * speed
            self.vy = math.sin(angle) * speed
            self.lifetime = self.rng.integers(15, 30)
            self.size = self.rng.integers(2, 5)

        def update(self):
            self.x += self.vx
            self.y += self.vy
            self.lifetime -= 1
            self.vx *= 0.95
            self.vy *= 0.95
            self.size = max(0, self.size - 0.1)

        def is_alive(self):
            return self.lifetime > 0

        def draw(self, surface):
            if self.is_alive():
                alpha = int(255 * (self.lifetime / 30))
                temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, self.color + (alpha,), (self.size, self.size), self.size)
                surface.blit(temp_surf, (self.x - self.size, self.y - self.size), special_flags=pygame.BLEND_RGBA_ADD)

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Asteroid Grid Shooter")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # none
        space = 0 # released
        shift = 0 # released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        
        # In a real-time game, you'd step every frame.
        # Here, we only step if an action is taken to match the turn-based nature.
        # This makes it playable for a human.
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
            if terminated:
                print("Game Over! Resetting in 3 seconds...")
                pygame.time.wait(3000)
                obs, info = env.reset()

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(10) # Limit frame rate for human playability

    pygame.quit()