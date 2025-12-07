
# Generated: 2025-08-27T17:56:39.217542
# Source Brief: brief_01684.md
# Brief Index: 1684

        
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
        "Controls: Arrow keys to move the cursor. Press space to place a block."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Build a fortress of blocks to guide falling orbs into collectors, "
        "maximizing your score while preventing orbs from escaping."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self._define_constants()

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
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        
        # Etc...        
        self.rng = np.random.default_rng()
        
        # Initialize state variables
        self.reset()

    def _define_constants(self):
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        
        self.CELL_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.CELL_SIZE

        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_BLOCK = (120, 130, 150)
        self.COLOR_ORB = (255, 255, 255)
        self.COLOR_CURSOR = (100, 255, 100, 100)
        self.COLOR_COLLECTOR = (255, 215, 0)
        self.COLOR_ESCAPE_LINE = (200, 50, 50)
        self.COLOR_TEXT = (230, 230, 230)

        # Game parameters
        self.ORB_RADIUS = 5
        self.ORB_SPEED = 3
        self.MAX_ESCAPED_ORBS = 5
        self.WIN_SCORE = 100
        self.MAX_STEPS = 3000
        
        self.BASE_SPAWN_INTERVAL = 2 * self.FPS # 2 seconds
        self.SPAWN_INTERVAL_DECREMENT = 0.1 * self.FPS # 0.1 seconds

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.cursor_pos = pygame.Vector2(self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        
        self.orbs = []
        self.particles = []
        
        self.score = 0
        self.escaped_orbs = 0
        self.steps = 0
        self.game_over = False
        
        self.orb_spawn_timer = self.BASE_SPAWN_INTERVAL
        self.orb_spawn_interval = self.BASE_SPAWN_INTERVAL
        
        self.prev_space_held = False
        
        # Define collectors
        collector_width_cells = 4
        spacing_cells = (self.GRID_WIDTH - 3 * collector_width_cells) // 4
        self.collectors = []
        for i in range(3):
            x = (spacing_cells * (i + 1) + collector_width_cells * i) * self.CELL_SIZE
            self.collectors.append(pygame.Rect(x, self.HEIGHT - self.CELL_SIZE, collector_width_cells * self.CELL_SIZE, self.CELL_SIZE))

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        reward = 0
        
        block_placed = self._handle_input(movement, space_held)
        if block_placed:
            reward -= 0.5 # Small penalty for using resources

        self._spawn_orbs()
        
        pre_move_orb_info = {id(orb): self._get_dist_to_nearest_collector(orb['pos']) for orb in self.orbs}
        
        self._move_orbs()
        
        for orb in self.orbs:
            old_dist = pre_move_orb_info.get(id(orb))
            if old_dist is not None:
                new_dist = self._get_dist_to_nearest_collector(orb['pos'])
                reward += (old_dist - new_dist) * 0.01

        collection_reward, escape_penalty = self._check_orb_outcomes()
        reward += collection_reward + escape_penalty

        self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over:
            if self.score >= self.WIN_SCORE:
                reward += 100
            if self.escaped_orbs >= self.MAX_ESCAPED_ORBS:
                reward -= 100
        
        self.game_over = terminated

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        if movement == 1: # Up
            self.cursor_pos.y = max(0, self.cursor_pos.y - 1)
        elif movement == 2: # Down
            self.cursor_pos.y = min(self.GRID_HEIGHT - 1, self.cursor_pos.y + 1)
        elif movement == 3: # Left
            self.cursor_pos.x = max(0, self.cursor_pos.x - 1)
        elif movement == 4: # Right
            self.cursor_pos.x = min(self.GRID_WIDTH - 1, self.cursor_pos.x + 1)
        
        block_placed = False
        if space_held and not self.prev_space_held:
            cx, cy = int(self.cursor_pos.x), int(self.cursor_pos.y)
            if self.grid[cx, cy] == 0 and cy > 0 and cy < self.GRID_HEIGHT - 1:
                self.grid[cx, cy] = 1
                block_placed = True
                # sfx: block_place.wav
        self.prev_space_held = space_held
        return block_placed

    def _spawn_orbs(self):
        self.orb_spawn_timer -= 1
        if self.orb_spawn_timer <= 0:
            spawn_x = self.rng.integers(self.ORB_RADIUS, self.WIDTH - self.ORB_RADIUS)
            self.orbs.append({
                "pos": pygame.Vector2(spawn_x, self.ORB_RADIUS),
                "vel": pygame.Vector2(0, self.ORB_SPEED)
            })
            self.orb_spawn_timer = self.orb_spawn_interval
            # sfx: orb_spawn.wav

    def _move_orbs(self):
        for orb in self.orbs:
            original_pos = orb['pos'].copy()
            # Move on X axis
            orb['pos'].x += orb['vel'].x
            if self._collides_with_block(orb) or orb['pos'].x < self.ORB_RADIUS or orb['pos'].x > self.WIDTH - self.ORB_RADIUS:
                orb['pos'].x = original_pos.x
                orb['vel'].x *= -1
                # sfx: bounce.wav
            # Move on Y axis
            orb['pos'].y += orb['vel'].y
            if self._collides_with_block(orb):
                orb['pos'].y = original_pos.y
                orb['vel'].y *= -1
                # sfx: bounce.wav

    def _collides_with_block(self, orb):
        orb_rect = pygame.Rect(orb['pos'].x - self.ORB_RADIUS, orb['pos'].y - self.ORB_RADIUS, self.ORB_RADIUS * 2, self.ORB_RADIUS * 2)
        gx_min = max(0, (int(orb_rect.left) // self.CELL_SIZE) - 1)
        gx_max = min(self.GRID_WIDTH, (int(orb_rect.right) // self.CELL_SIZE) + 1)
        gy_min = max(0, (int(orb_rect.top) // self.CELL_SIZE) - 1)
        gy_max = min(self.GRID_HEIGHT, (int(orb_rect.bottom) // self.CELL_SIZE) + 1)
        for gx in range(gx_min, gx_max):
            for gy in range(gy_min, gy_max):
                if self.grid[gx, gy] == 1:
                    block_rect = pygame.Rect(gx * self.CELL_SIZE, gy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                    if block_rect.colliderect(orb_rect):
                        return True
        return False

    def _check_orb_outcomes(self):
        reward = 0
        penalty = 0
        orbs_to_remove = []
        for orb in self.orbs:
            collected = False
            for collector in self.collectors:
                if collector.collidepoint(orb['pos']):
                    self.score += 1
                    reward += 10
                    orbs_to_remove.append(orb)
                    self._create_particles(orb['pos'])
                    collected = True
                    # sfx: collect.wav
                    difficulty_level = self.score // 50
                    self.orb_spawn_interval = max(self.FPS, self.BASE_SPAWN_INTERVAL - difficulty_level * self.SPAWN_INTERVAL_DECREMENT)
                    break
            if collected: continue
            if orb['pos'].y > self.HEIGHT:
                self.escaped_orbs += 1
                penalty -= 2
                orbs_to_remove.append(orb)
                # sfx: escape.wav
        self.orbs = [orb for orb in self.orbs if orb not in orbs_to_remove]
        return reward, penalty

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_dist_to_nearest_collector(self, pos):
        if not self.collectors: return float('inf')
        return min(pos.distance_to(c.center) for c in self.collectors)

    def _create_particles(self, pos):
        for _ in range(20):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 2 + 1
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                "life": self.rng.integers(15, 30),
                "color": self.COLOR_COLLECTOR
            })
            
    def _check_termination(self):
        return (
            self.score >= self.WIN_SCORE or
            self.escaped_orbs >= self.MAX_ESCAPED_ORBS or
            self.steps >= self.MAX_STEPS
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Grid
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
        # Collectors
        for collector in self.collectors:
            glow_rect = collector.inflate(10, 10)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_COLLECTOR, 50), s.get_rect(), border_radius=5)
            self.screen.blit(s, glow_rect.topleft)
            pygame.draw.rect(self.screen, self.COLOR_COLLECTOR, collector, border_radius=3)
        # Escape line
        pygame.draw.line(self.screen, self.COLOR_ESCAPE_LINE, (0, self.HEIGHT - 1), (self.WIDTH, self.HEIGHT - 1), 2)
        # Blocks
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] == 1:
                    rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                    pygame.draw.rect(self.screen, self.COLOR_BLOCK, rect)
                    pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)
        # Orbs
        for orb in self.orbs:
            pos = (int(orb['pos'].x), int(orb['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ORB_RADIUS, self.COLOR_ORB)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ORB_RADIUS, self.COLOR_ORB)
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            color = (*p['color'], alpha)
            pos = (int(p['pos'].x), int(p['pos'].y))
            size = max(0, int(2 * (p['life'] / 30.0)))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)
        # Cursor
        cursor_rect = pygame.Rect(self.cursor_pos.x * self.CELL_SIZE, self.cursor_pos.y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        s.fill(self.COLOR_CURSOR)
        self.screen.blit(s, cursor_rect.topleft)
        pygame.draw.rect(self.screen, (255,255,255), cursor_rect, 1)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        escaped_text = self.font.render(f"ESCAPED: {self.escaped_orbs}/{self.MAX_ESCAPED_ORBS}", True, self.COLOR_TEXT)
        self.screen.blit(escaped_text, (self.WIDTH - escaped_text.get_width() - 10, 10))
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            status_text = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            status_render = self.font.render(status_text, True, self.COLOR_COLLECTOR if self.score >= self.WIN_SCORE else self.COLOR_ESCAPE_LINE)
            status_rect = status_render.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(status_render, status_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "escaped_orbs": self.escaped_orbs,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Orb Fortress")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(env.FPS)
        
    env.close()