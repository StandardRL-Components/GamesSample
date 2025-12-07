
# Generated: 2025-08-28T03:53:49.295312
# Source Brief: brief_02153.md
# Brief Index: 2153

        
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
        "Controls: Arrow keys to move in isometric directions. Collect all the gems before time runs out!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate an isometric maze, collecting gems while dodging traps to achieve a high score before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    COLOR_BG = (25, 28, 44)
    COLOR_GRID_LINE = (45, 50, 70)
    COLOR_PLAYER = (255, 204, 0)
    COLOR_PLAYER_GLOW = (255, 204, 0, 50)
    COLOR_TRAP = (50, 50, 60)
    COLOR_TRAP_WARN = (255, 50, 50)
    COLOR_TEXT = (230, 230, 230)
    COLOR_TEXT_SHADOW = (20, 20, 20)
    GEM_COLORS = [
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 0),    # Lime
        (255, 128, 0),  # Orange
    ]

    GRID_WIDTH, GRID_HEIGHT = 40, 40
    TILE_WIDTH = 48
    TILE_HEIGHT = 24
    TILE_WIDTH_HALF = TILE_WIDTH // 2
    TILE_HEIGHT_HALF = TILE_HEIGHT // 2

    TIME_LIMIT = 600
    MAX_EPISODE_STEPS = 1000
    GEM_TARGET = 50
    NUM_TRAPS = 75

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Etc...        
        # Initialize state variables
        self.player_pos = pygame.Vector2(0, 0)
        self.camera_pos = pygame.Vector2(0, 0)
        self.target_camera_pos = pygame.Vector2(0, 0)
        self.gems = []
        self.traps = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False
        self.last_gem_dist = float('inf')
        self.last_trap_dist = float('inf')

        self.reset()
        # self.validate_implementation()
    
    def _iso_to_screen(self, x, y):
        screen_x = (x - y) * self.TILE_WIDTH_HALF
        screen_y = (x + y) * self.TILE_HEIGHT_HALF
        return pygame.Vector2(screen_x, screen_y)

    def _generate_map(self):
        start_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.player_pos = pygame.Vector2(start_pos)

        # Use BFS to find all reachable tiles to ensure a connected map
        queue = [start_pos]
        visited = {start_pos}
        
        while queue:
            x, y = queue.pop(0)
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        possible_locations = list(visited - {start_pos})
        self.np_random.shuffle(possible_locations)
        
        # Place gems
        num_gems = min(self.GEM_TARGET, len(possible_locations))
        self.gems = [pygame.Vector2(pos) for pos in possible_locations[:num_gems]]
        
        # Place traps
        remaining_locations = possible_locations[num_gems:]
        num_traps = min(self.NUM_TRAPS, len(remaining_locations))
        self.traps = [pygame.Vector2(pos) for pos in remaining_locations[:num_traps]]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_map()
        
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False
        self.particles = []

        # Center camera on player
        self.target_camera_pos = self._iso_to_screen(self.player_pos.x, self.player_pos.y)
        self.camera_pos = self.target_camera_pos.copy()

        # Pre-calculate initial distances for reward
        self.last_gem_dist, _ = self._get_closest_entity(self.gems)
        self.last_trap_dist, _ = self._get_closest_entity(self.traps)

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        self.game_over = False

        prev_pos = self.player_pos.copy()

        if movement == 1: # Up-Left
            self.player_pos.y -= 1
        elif movement == 2: # Down-Right
            self.player_pos.y += 1
        elif movement == 3: # Down-Left
            self.player_pos.x -= 1
        elif movement == 4: # Up-Right
            self.player_pos.x += 1
        
        # Clamp player position to grid
        self.player_pos.x = max(0, min(self.GRID_WIDTH - 1, self.player_pos.x))
        self.player_pos.y = max(0, min(self.GRID_HEIGHT - 1, self.player_pos.y))

        self.steps += 1

        # --- Collision and Events ---
        if self.player_pos in self.gems:
            self.gems.remove(self.player_pos)
            self.gems_collected += 1
            self.score += 10
            reward += 10
            # SFX: Gem collect sound
            self._create_particles(self.player_pos, random.choice(self.GEM_COLORS))

        if self.player_pos in self.traps:
            self.game_over = True
            self.score -= 50
            reward = -50
            # SFX: Player fall/trap sound

        # --- Continuous Reward Shaping ---
        if not self.game_over:
            gem_dist, _ = self._get_closest_entity(self.gems)
            trap_dist, _ = self._get_closest_entity(self.traps)

            if gem_dist < self.last_gem_dist:
                reward += 1.0
            elif movement != 0:
                reward -= 1.0

            if trap_dist < self.last_trap_dist:
                reward -= 0.1
            
            self.last_gem_dist = gem_dist
            self.last_trap_dist = trap_dist

        # --- Termination Conditions ---
        terminated = False
        if self.game_over:
            terminated = True
        elif self.gems_collected >= self.GEM_TARGET:
            terminated = True
            self.score += 50
            reward += 50
            # SFX: Victory fanfare
        elif self.steps >= self.TIME_LIMIT:
            terminated = True
        elif self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True
        
        # --- Update Camera ---
        self.target_camera_pos = self._iso_to_screen(self.player_pos.x, self.player_pos.y)
        # Smooth camera movement
        self.camera_pos.x += (self.target_camera_pos.x - self.camera_pos.x) * 0.1
        self.camera_pos.y += (self.target_camera_pos.y - self.camera_pos.y) * 0.1
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_closest_entity(self, entity_list):
        if not entity_list:
            return float('inf'), None
        
        closest_dist = float('inf')
        closest_entity = None
        for entity in entity_list:
            dist = self.player_pos.distance_to(entity)
            if dist < closest_dist:
                closest_dist = dist
                closest_entity = entity
        return closest_dist, closest_entity

    def _create_particles(self, pos, color):
        screen_pos = self._iso_to_screen(pos.x, pos.y)
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            velocity = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': screen_pos.copy(),
                'vel': velocity,
                'life': random.randint(15, 30),
                'color': color
            })

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
        # Calculate camera offset
        offset = pygame.Vector2(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 4) - self.camera_pos

        # Determine visible grid range
        view_range_x = int(self.SCREEN_WIDTH / self.TILE_WIDTH) + 4
        view_range_y = int(self.SCREEN_HEIGHT / self.TILE_HEIGHT) + 4
        
        center_grid_x = int(self.player_pos.x)
        center_grid_y = int(self.player_pos.y)

        # Draw grid lines
        for y in range(max(0, center_grid_y - view_range_y), min(self.GRID_HEIGHT, center_grid_y + view_range_y)):
            for x in range(max(0, center_grid_x - view_range_x), min(self.GRID_WIDTH, center_grid_x + view_range_x)):
                p = self._iso_to_screen(x, y) + offset
                points = [
                    (p.x, p.y + self.TILE_HEIGHT_HALF),
                    (p.x + self.TILE_WIDTH_HALF, p.y),
                    (p.x, p.y - self.TILE_HEIGHT_HALF),
                    (p.x - self.TILE_WIDTH_HALF, p.y),
                ]
                pygame.gfxdraw.aapolygon(self.screen, [(int(px), int(py)) for px, py in points], self.COLOR_GRID_LINE)
        
        # Draw Traps
        for trap_pos in self.traps:
            p = self._iso_to_screen(trap_pos.x, trap_pos.y) + offset
            if -self.TILE_WIDTH < p.x < self.SCREEN_WIDTH + self.TILE_WIDTH and -self.TILE_HEIGHT < p.y < self.SCREEN_HEIGHT + self.TILE_HEIGHT:
                is_adjacent = self.player_pos.distance_to(trap_pos) < 1.5
                flash_on = (self.steps % 10 < 5)
                color = self.COLOR_TRAP_WARN if (is_adjacent and flash_on) else self.COLOR_TRAP
                points = [
                    (p.x, p.y + self.TILE_HEIGHT_HALF * 0.8),
                    (p.x + self.TILE_WIDTH_HALF * 0.8, p.y),
                    (p.x, p.y - self.TILE_HEIGHT_HALF * 0.8),
                    (p.x - self.TILE_WIDTH_HALF * 0.8, p.y),
                ]
                pygame.gfxdraw.filled_polygon(self.screen, [(int(px), int(py)) for px, py in points], color)

        # Draw Gems
        for i, gem_pos in enumerate(self.gems):
            p = self._iso_to_screen(gem_pos.x, gem_pos.y) + offset
            if -self.TILE_WIDTH < p.x < self.SCREEN_WIDTH + self.TILE_WIDTH and -self.TILE_HEIGHT < p.y < self.SCREEN_HEIGHT + self.TILE_HEIGHT:
                angle = (self.steps * 4 + i * 30) * (math.pi / 180)
                size = self.TILE_WIDTH_HALF * 0.5
                color = self.GEM_COLORS[i % len(self.GEM_COLORS)]
                points = []
                for j in range(4):
                    a = angle + j * (math.pi / 2)
                    points.append((p.x + math.cos(a) * size, p.y + math.sin(a) * size * 0.5))
                pygame.gfxdraw.filled_polygon(self.screen, [(int(px), int(py)) for px, py in points], color)
                pygame.gfxdraw.aapolygon(self.screen, [(int(px), int(py)) for px, py in points], (255,255,255))

        # Render Particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                render_pos = p['pos'] + offset
                radius = max(0, int(p['life'] / 6))
                pygame.draw.circle(self.screen, p['color'], (int(render_pos.x), int(render_pos.y)), radius)

        # Render Player
        player_screen_pos = self._iso_to_screen(self.player_pos.x, self.player_pos.y) + offset
        bob_offset = math.sin(self.steps * 0.3) * 3
        player_y = player_screen_pos.y - self.TILE_HEIGHT_HALF * 0.7 + bob_offset
        
        glow_surf = pygame.Surface((self.TILE_WIDTH, self.TILE_WIDTH), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (self.TILE_WIDTH // 2, self.TILE_WIDTH // 2), self.TILE_WIDTH // 2)
        self.screen.blit(glow_surf, (int(player_screen_pos.x - self.TILE_WIDTH // 2), int(player_y - self.TILE_WIDTH // 2 + 5)))

        points = [
            (player_screen_pos.x, player_y - self.TILE_HEIGHT_HALF * 0.8),
            (player_screen_pos.x + self.TILE_WIDTH_HALF * 0.5, player_y),
            (player_screen_pos.x, player_y + self.TILE_HEIGHT_HALF * 0.8),
            (player_screen_pos.x - self.TILE_WIDTH_HALF * 0.5, player_y),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, [(int(px), int(py)) for px, py in points], self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, [(int(px), int(py)) for px, py in points], (255, 255, 255))

    def _render_text(self, text, font, position, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, align="center"):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        
        if align == "center":
            text_rect = text_surf.get_rect(center=position)
            shadow_rect = shadow_surf.get_rect(center=(position[0] + 2, position[1] + 2))
        elif align == "topleft":
            text_rect = text_surf.get_rect(topleft=position)
            shadow_rect = shadow_surf.get_rect(topleft=(position[0] + 2, position[1] + 2))
        elif align == "topright":
            text_rect = text_surf.get_rect(topright=position)
            shadow_rect = shadow_surf.get_rect(topright=(position[0] + 2, position[1] + 2))

        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        self._render_text(f"Score: {self.score}", self.font_large, (20, 15), align="topleft")
        time_left = max(0, self.TIME_LIMIT - self.steps)
        self._render_text(f"Time: {time_left}", self.font_large, (self.SCREEN_WIDTH - 20, 15), align="topright")
        self._render_text(f"Gems: {self.gems_collected} / {self.GEM_TARGET}", self.font_large, (self.SCREEN_WIDTH // 2, 30))

        if self.game_over:
            self._render_text("GAME OVER", pygame.font.Font(None, 80), (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20))
        elif self.gems_collected >= self.GEM_TARGET:
            self._render_text("YOU WIN!", pygame.font.Font(None, 80), (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_collected": self.gems_collected,
            "player_pos": (self.player_pos.x, self.player_pos.y),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        assert "score" in info
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        assert "steps" in info

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Isometric Gem Collector")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    last_action_time = pygame.time.get_ticks()
    key_repeat_delay = 100 # ms

    while running:
        action = [0, 0, 0] # Default action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                done = False

        if not done:
            keys = pygame.key.get_pressed()
            movement = 0 
            
            # Allow key holding for continuous movement
            current_time = pygame.time.get_ticks()
            if current_time - last_action_time > key_repeat_delay:
                if keys[pygame.K_UP]: movement = 1
                elif keys[pygame.K_DOWN]: movement = 2
                elif keys[pygame.K_LEFT]: movement = 3
                elif keys[pygame.K_RIGHT]: movement = 4
                
                if movement != 0:
                    action[0] = movement
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                    last_action_time = current_time

        # Render the current state
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) 

    env.close()