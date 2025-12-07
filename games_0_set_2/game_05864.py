import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
from collections import deque
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move in isometric directions. Your goal is to collect 10 gems and reach the blue exit portal."
    )

    game_description = (
        "Navigate the procedurally generated Crystal Caverns, collecting gems and dodging pulsating traps to reach the exit."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 40, 40
        self.MAX_STEPS = 1000
        self.MIN_GEMS_FOR_EXIT = 10
        self.INITIAL_HEALTH = 3

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # Visuals
        self._init_colors()
        self._init_fonts()
        self._init_isometric_params()
        
        # State variables (will be initialized in reset)
        self.grid = None
        self.player_pos = None
        self.exit_pos = None
        self.gems = None
        self.traps = None
        self.steps = 0
        self.score = 0
        self.health = 0
        self.gem_count = 0
        self.game_over = False
        self.win = False
        self.last_reward = 0
        self.trap_cycle_duration = 5
        self.trap_timer = 0
        self.particles = []
        self.np_random = None

        # self.reset() is called by the wrapper, no need to call it here.
        # self.validate_implementation() # Not needed in the final version

    def _init_colors(self):
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_FLOOR = (40, 45, 70)
        self.COLOR_FLOOR_OUTLINE = (60, 65, 90)
        self.COLOR_WALL_TOP = (90, 95, 130)
        self.COLOR_WALL_SIDE = (70, 75, 110)
        self.COLOR_PLAYER = (50, 255, 255)
        self.COLOR_PLAYER_GLOW = (50, 255, 255, 50)
        self.COLOR_EXIT = (100, 100, 255)
        self.COLOR_EXIT_GLOW = (100, 100, 255, 100)
        self.GEM_COLORS = [(255, 80, 80), (80, 255, 80), (80, 80, 255)]
        self.COLOR_TRAP_INACTIVE = (50, 50, 50)
        self.COLOR_TRAP_ACTIVE = (255, 120, 0)
        self.COLOR_TRAP_GLOW = (255, 120, 0, 150)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_SUCCESS = (100, 255, 100)
        self.COLOR_UI_FAIL = (255, 100, 100)

    def _init_fonts(self):
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 64)

    def _init_isometric_params(self):
        self.TILE_WIDTH_HALF = 16
        self.TILE_HEIGHT_HALF = 8
        self.WALL_HEIGHT = 20
        self.origin_x = self.WIDTH // 2
        self.origin_y = 60

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.health = self.INITIAL_HEALTH
        self.gem_count = 0
        self.game_over = False
        self.win = False
        self.last_reward = 0
        self.trap_cycle_duration = 5
        self.trap_timer = 0
        self.particles = []

        self._generate_cavern()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, _, _ = action
        reward = -0.1  # Cost of living

        # Update game logic
        self.steps += 1
        
        # Update trap cycle
        if self.steps > 0 and self.steps % 200 == 0:
            self.trap_cycle_duration = max(2, self.trap_cycle_duration - 1)
        self.trap_timer = (self.trap_timer + 1) % self.trap_cycle_duration
        is_trap_active = self.trap_timer == 0

        # Handle player movement
        if movement != 0:
            dx, dy = [(0,0), (0,-1), (0,1), (-1,0), (1,0)][movement]
            new_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            
            if 0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT and self.grid[new_pos[1]][new_pos[0]] == 0:
                self.player_pos = new_pos

        # Check for interactions
        if self.player_pos in self.gems:
            self.gems.remove(self.player_pos)
            self.gem_count += 1
            reward += 1.0
            # sfx: gem collect
            gem_color = self.GEM_COLORS[self.np_random.integers(len(self.GEM_COLORS))]
            self._create_particles(self.player_pos, gem_color, 10)

        if is_trap_active and self.player_pos in self.traps:
            self.health -= 1
            reward -= 5.0
            # sfx: player hurt
            self._create_particles(self.player_pos, self.COLOR_TRAP_ACTIVE, 20, is_explosion=True)

        self.last_reward = reward
        self.score += reward
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if self.win:
            reward += 50.0
            self.score += 50.0

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _check_termination(self):
        if self.health <= 0:
            self.game_over = True
            return True
        if self.player_pos == self.exit_pos and self.gem_count >= self.MIN_GEMS_FOR_EXIT:
            self.game_over = True
            self.win = True
            return True
        return False

    def _grid_to_iso(self, x, y):
        iso_x = self.origin_x + (x - y) * self.TILE_WIDTH_HALF
        iso_y = self.origin_y + (x + y) * self.TILE_HEIGHT_HALF
        return int(iso_x), int(iso_y)

    def _generate_cavern(self):
        self.grid = np.ones((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        start_pos = (self.np_random.integers(2, 8), self.np_random.integers(2, 8))
        
        def get_neighbors(pos):
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = pos[0] + dx, pos[1] + dy
                if 1 <= nx < self.GRID_WIDTH - 1 and 1 <= ny < self.GRID_HEIGHT - 1:
                    neighbors.append((nx, ny))
            return neighbors

        # Carve a main path using random walk with a goal bias
        path = [start_pos]
        curr = start_pos
        self.grid[curr[1]][curr[0]] = 0
        for _ in range(200):
            neighbors = get_neighbors(curr)
            weights = [3 if n[0] > curr[0] or n[1] > curr[1] else 1 for n in neighbors]
            if not neighbors: break
            
            # FIX: np.random.Generator.choice with a 2D-like 'a' and 'p' is problematic.
            # Instead, choose an index with weights, then select the element.
            # This also ensures 'curr' is a tuple, not a numpy array, fixing the ValueError.
            chosen_index = self.np_random.choice(len(neighbors), p=np.array(weights)/sum(weights))
            curr = neighbors[chosen_index]
            
            self.grid[curr[1]][curr[0]] = 0
            if curr not in path: path.append(curr)

        # Carve more open spaces
        for _ in range(10):
            walker_pos = path[self.np_random.integers(0, len(path))]
            for _ in range(50):
                neighbors = get_neighbors(walker_pos)
                if not neighbors: break
                # FIX: Ensure walker_pos is a tuple for consistency.
                walker_pos = neighbors[self.np_random.integers(len(neighbors))]
                self.grid[walker_pos[1]][walker_pos[0]] = 0
        
        # Set player start and exit
        self.player_pos = start_pos
        self.exit_pos = curr
        
        # Place entities
        floor_tiles = np.argwhere(self.grid == 0)
        self.np_random.shuffle(floor_tiles)
        floor_tiles = [(c, r) for r, c in floor_tiles]
        
        safe_path_for_traps = set(path[:50])
        
        self.gems = set()
        self.traps = set()
        
        gem_count = 0
        trap_count = 0
        
        for pos in floor_tiles:
            if pos == self.player_pos or pos == self.exit_pos:
                continue
            
            if gem_count < 25 and self.np_random.random() < 0.08:
                self.gems.add(pos)
                gem_count += 1
            elif trap_count < 15 and pos not in safe_path_for_traps and self.np_random.random() < 0.05:
                self.traps.add(pos)
                trap_count += 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_game()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Create a list of all drawable objects
        draw_list = []
        is_trap_active = self.trap_timer == 0
        
        # Add tiles
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                sort_key = c + r
                if self.grid[r][c] == 1: # Wall
                    draw_list.append(('wall', (c, r), sort_key))
                else: # Floor
                    draw_list.append(('floor', (c, r), sort_key))
                    if (c,r) in self.gems:
                        draw_list.append(('gem', (c,r), sort_key + 0.5))
                    if (c,r) in self.traps:
                        # Pass the current trap state, not the object itself
                        draw_list.append(('trap', (c,r), is_trap_active, sort_key + 0.1))
                    if (c,r) == self.exit_pos:
                        draw_list.append(('exit', (c,r), sort_key + 0.2))

        # Add player
        draw_list.append(('player', self.player_pos, self.player_pos[0] + self.player_pos[1] + 0.6))
        
        # Sort and draw
        draw_list.sort(key=lambda item: item[2])
        
        for item in draw_list:
            obj_type = item[0]
            pos = item[1]
            iso_x, iso_y = self._grid_to_iso(pos[0], pos[1])
            
            if obj_type == 'floor':
                self._draw_iso_tile(iso_x, iso_y, self.COLOR_FLOOR, self.COLOR_FLOOR_OUTLINE)
            elif obj_type == 'wall':
                self._draw_iso_wall(iso_x, iso_y)
            elif obj_type == 'gem':
                self._draw_gem(iso_x, iso_y, pos)
            elif obj_type == 'trap':
                is_active = item[2] # The state was passed in the list
                self._draw_trap(iso_x, iso_y, is_active)
            elif obj_type == 'exit':
                self._draw_exit(iso_x, iso_y)
            elif obj_type == 'player':
                self._draw_player(iso_x, iso_y)

    def _draw_iso_tile(self, x, y, color, outline_color):
        points = [
            (x, y - self.TILE_HEIGHT_HALF),
            (x + self.TILE_WIDTH_HALF, y),
            (x, y + self.TILE_HEIGHT_HALF),
            (x - self.TILE_WIDTH_HALF, y)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, outline_color)

    def _draw_iso_wall(self, x, y):
        top_points = [
            (x, y - self.TILE_HEIGHT_HALF - self.WALL_HEIGHT),
            (x + self.TILE_WIDTH_HALF, y - self.WALL_HEIGHT),
            (x, y + self.TILE_HEIGHT_HALF - self.WALL_HEIGHT),
            (x - self.TILE_WIDTH_HALF, y - self.WALL_HEIGHT)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, top_points, self.COLOR_WALL_TOP)
        pygame.gfxdraw.aapolygon(self.screen, top_points, self.COLOR_WALL_SIDE)

        # Side faces
        left_face = [
            (x - self.TILE_WIDTH_HALF, y - self.WALL_HEIGHT),
            (x, y + self.TILE_HEIGHT_HALF - self.WALL_HEIGHT),
            (x, y + self.TILE_HEIGHT_HALF),
            (x - self.TILE_WIDTH_HALF, y)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, left_face, self.COLOR_WALL_SIDE)
        pygame.gfxdraw.aapolygon(self.screen, left_face, self.COLOR_WALL_SIDE)
        
        right_face = [
            (x + self.TILE_WIDTH_HALF, y - self.WALL_HEIGHT),
            (x, y + self.TILE_HEIGHT_HALF - self.WALL_HEIGHT),
            (x, y + self.TILE_HEIGHT_HALF),
            (x + self.TILE_WIDTH_HALF, y)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, right_face, self.COLOR_WALL_SIDE)
        pygame.gfxdraw.aapolygon(self.screen, right_face, self.COLOR_WALL_SIDE)

    def _draw_gem(self, x, y, pos):
        color_index = (pos[0] + pos[1]) % len(self.GEM_COLORS)
        color = self.GEM_COLORS[color_index]
        
        glow_size = 5 + 2 * (math.sin(self.steps * 0.1 + pos[0]) + 1)
        glow_color = (*color, 70)
        pygame.gfxdraw.filled_circle(self.screen, x, y - 8, int(glow_size), glow_color)
        
        points = [
            (x, y - 12), (x + 4, y - 8), (x, y - 4), (x - 4, y - 8)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, (255,255,255))

    def _draw_trap(self, x, y, is_active):
        if is_active:
            pulse = (math.sin(pygame.time.get_ticks() * 0.02) + 1) / 2
            radius = int(6 + pulse * 4)
            pygame.gfxdraw.filled_circle(self.screen, x, y, radius, self.COLOR_TRAP_ACTIVE)
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, self.COLOR_TRAP_GLOW)
            for i in range(8):
                angle = i * math.pi / 4 + self.steps * 0.1
                sx = x + math.cos(angle) * (radius - 2)
                ex = x + math.cos(angle) * (radius + 2)
                sy = y + math.sin(angle) * (radius - 2)
                ey = y + math.sin(angle) * (radius + 2)
                pygame.draw.aaline(self.screen, self.COLOR_TRAP_GLOW, (sx, sy), (ex, ey))
        else:
            pygame.gfxdraw.filled_circle(self.screen, x, y, 6, self.COLOR_TRAP_INACTIVE)

    def _draw_exit(self, x, y):
        time = pygame.time.get_ticks() * 0.002
        for i in range(4):
            radius = (10 + i * 3 + math.sin(time + i) * 2)
            alpha = int(100 + math.sin(time + i) * 50)
            color = (*self.COLOR_EXIT, alpha)
            pygame.gfxdraw.aacircle(self.screen, x, y - 10, int(radius), color)

    def _draw_player(self, x, y):
        bob = math.sin(self.steps * 0.4) * 3
        y_pos = y - 12 + int(bob)
        
        pygame.gfxdraw.filled_circle(self.screen, x, y_pos, 10, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, x, y_pos, 6, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, x, y_pos, 6, (200, 255, 255))
        
    def _create_particles(self, grid_pos, color, count, is_explosion=False):
        iso_x, iso_y = self._grid_to_iso(grid_pos[0], grid_pos[1])
        for _ in range(count):
            if is_explosion:
                angle = self.np_random.random() * 2 * math.pi
                speed = 1 + self.np_random.random() * 2
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            else: # Gem collection effect
                vel = [(self.np_random.random() - 0.5) * 2, -1 - self.np_random.random() * 2]

            self.particles.append({
                'pos': [iso_x, iso_y - 10],
                'vel': vel,
                'life': 20 + self.np_random.integers(0, 10),
                'color': color,
                'radius': 2 + self.np_random.random() * 2
            })

    def _render_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            
            if p['life'] <= 0:
                self.particles.remove(p)
                continue
            
            radius = p['radius'] * (p['life'] / 30)
            if radius > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(radius), p['color'])

    def _render_ui(self):
        health_text = self.font_ui.render(f"Health: {self.health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10, 10))
        
        gem_color = self.COLOR_UI_SUCCESS if self.gem_count >= self.MIN_GEMS_FOR_EXIT else self.COLOR_UI_TEXT
        gem_text = self.font_ui.render(f"Gems: {self.gem_count} / {self.MIN_GEMS_FOR_EXIT}", True, gem_color)
        self.screen.blit(gem_text, (10, 30))

        steps_text = self.font_ui.render(f"Steps: {self.steps} / {self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 30))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                end_text = self.font_game_over.render("VICTORY!", True, self.COLOR_UI_SUCCESS)
            else:
                end_text = self.font_game_over.render("GAME OVER", True, self.COLOR_UI_FAIL)
            
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.health,
            "gem_count": self.gem_count,
            "win": self.win,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Make sure to unset the dummy video driver if you want to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Crystal Caverns")
    clock = pygame.time.Clock()
    
    running = True
    total_score = 0
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_score = 0
                    print("--- Game Reset ---")
                
                # Map arrow keys to isometric movement
                # Note: This mapping is for intuitive play, not necessarily optimal
                # It assumes a specific isometric projection
                if keys[pygame.K_UP]:
                    action[0] = 1 # Up-Left in iso
                elif keys[pygame.K_DOWN]:
                    action[0] = 2 # Down-Right in iso
                elif keys[pygame.K_LEFT]:
                    action[0] = 3 # Down-Left in iso
                elif keys[pygame.K_RIGHT]:
                    action[0] = 4 # Up-Right in iso

        # The game should advance even if no key is pressed, to update animations/traps
        # The original code only stepped on key press, this is a better way for interactive play
        keys = pygame.key.get_pressed()
        move_made = False
        if keys[pygame.K_UP]:
            action[0] = 1; move_made = True
        elif keys[pygame.K_DOWN]:
            action[0] = 2; move_made = True
        elif keys[pygame.K_LEFT]:
            action[0] = 3; move_made = True
        elif keys[pygame.K_RIGHT]:
            action[0] = 4; move_made = True

        obs, reward, terminated, truncated, info = env.step(action)
        total_score += reward
        if move_made:
             print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Score: {total_score:.2f}, Terminated: {terminated or truncated}")

        if terminated or truncated:
            print("--- Game Over ---")
            print(f"Final Score: {info['score']:.2f}, Win: {info['win']}")
            # Wait for reset
            
        # Render the environment observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate
        
    env.close()